# Plan : prix d'exécution piloté par signal (fill au niveau calculé)

Branche : à créer depuis `main` @ `eac22e6` (0.15.0).

## État d'avancement

| Phase | Statut |
|---|---|
| 0 : garde-fous perf | goldens verts avant/après ; bench à consigner |
| 1 : résolution du prix contre les signaux (boucles générales) | **fait** |
| 2 : surface utilisateur, validation, observabilité | **fait** |
| 3 : tests (dont repro utilisateur) | **fait** : 5 tests Rust + 3 tests Python (`test_exec_price_signal.py`, wheel debug dans un venv isolé) |
| 4 : mesure et documentation | doc + exemple runnable faits (`examples/21_fill_at_computed_level.py`, vérifié : mêmes 821 trades, -20.84% close vs +20.85% niveau) ; bench 0% AtClose TENTÉ 2026-08-17 : 5 passes alternées baseline/branche, variance de charge ~1.9x (branche mesurée 741µs ET 1.47ms, baseline 842µs ET 1.39ms, direction qui s'inverse entre paires) -- illisible sur machine active, À REFAIRE sur machine calme avant merge ; note : le bench (sans ordres, AtClose) passe par le kernel rapide où ce code ne s'exécute pas, une régression y serait structurelle, pas liée à ce chemin |

Arbitrages tranchés (2026-08-17) : `Custom` élargi (pas de variante nouvelle),
indexation `signal_row`, NaN = fallback close + warning.

But : permettre à `execution_price` de lire une **série calculée par la stratégie**
(signal du DSL), pas seulement une colonne du batch de barres. C'est ce qui manque à
toute stratégie « à niveau » (mean reversion sur bandes, pullback sur EMA, entrée sur
swing) : le moteur sait calculer le niveau, il ne sait pas remplir dessus. Le fill
tombe au close de la barre, systématiquement du mauvais côté sur un instrument qui
mean-reverse (mesuré sur le repro utilisateur : +19.09% au niveau contre -49.58% au
close, mêmes signaux, mêmes 293 ordres).

Contrainte cardinale, inchangée : **zéro coût quand la fonctionnalité n'est pas
utilisée**, et pas de nouvelle transcription de la règle de fill.

---

## 1. État des lieux

### Ce qui existe déjà

| Élément | Emplacement |
|---|---|
| `ExecutionPrice::Custom(String)` (serde + API Python `ExecutionPrice.custom()`) | `orchestrator.rs:298`, `helpers.py:125` |
| Résolution du prix, colonnes de barres uniquement | `fill.rs:58`, `Custom` à `fill.rs:89` via `f64_column(bars.batch, ...)` |
| Les DEUX seuls consommateurs : boucle générale de `run` et boucle générale lite | `orchestrator.rs:2017`, `orchestrator.rs:3336` |
| Récolte de séries par nom depuis les envs, alignement resample compris (0.15.0) | `collect_entry_series`, `orchestrator.rs:679` |
| SL/TP re-vérifiés sur la barre d'entrée quand le fill n'est pas au close | `check_entry_bar`, `orchestrator.rs:1755` |
| Warning quand un fill sort de `[low, high]` | `validate_fill_price`, `fill.rs:125` |
| Le kernel rapide et le GPU refusent déjà tout prix non-`AtClose` | `fast_path_blocker`, `orchestrator.rs:5285` |

Autrement dit : la sérialisation existe, la récolte de séries existe, le garde-fou
existe, l'interaction SL/TP est pensée, et le périmètre est confiné aux deux boucles
générales par construction. Il manque un seul fil : `Custom` ne regarde que
`bars.batch`.

### Les cinq impasses, mesurées (repro utilisateur, 0.15.0)

| Voie tentée | Résultat mesuré |
|---|---|
| Colonne supplémentaire dans les barres | supprimée à l'import |
| `ExecutionPrice.custom("prix_exec")` | ne voit pas les exo, colonnes de barres seulement |
| Loger le niveau dans `vwap` | écrasé par `(o+h+l+c)/4`, écart 0.0 exact |
| Loger le niveau dans `bid`/`ask` | jetés, 0 valeur non nulle |
| `.market_if_touched(signal=...)` | niveau gelé à la création de l'ordre (`limit_price: Option<f64>`) |

### Pourquoi PAS le repricing d'ordre au repos

Le besoin réel est un prix d'exécution par barre : le signal et le fill tombent sur la
même barre fine, le niveau est connu avant qu'elle commence. Le repricing d'un ordre
au repos est un chantier séparé (2 boucles + GPU + sémantique StopLimit + opt-in) qui
ne débloque même pas ce cas. Hors périmètre ici.

---

## 2. Conception

### 2.1 Résolution une fois par symbole, avant la boucle

```rust
/// Le prix d'exécution d'un symbole, résolu avant la boucle : soit le
/// comportement actuel (résolution par barre), soit une série de la stratégie.
enum ResolvedExecPrice<'a> {
    /// AtClose / AtOpen / AtVwap / MidPrice / Custom(colonne de barre) :
    /// comportement actuel, inchangé au bit près.
    PerBar(&'a ExecutionPrice),
    /// Custom(nom) résolu contre les signaux de la stratégie.
    Series(&'a [f64]),
}
```

Ordre de résolution de `Custom(nom)` :

1. colonne du batch de barres (rétro-compatibilité stricte, les configs actuelles ne
   bougent pas) ;
2. sinon, signal de la stratégie, récolté par la même mécanique que
   `collect_entry_series` (élargir la liste de noms passée à la récolte : noms
   d'entrée + nom du prix d'exécution, un seul appel) ;
3. sinon, **erreur avant la boucle** (aujourd'hui : erreur au premier fill, donc
   potentiellement des heures de simulation perdues avant de la voir) ;
4. collision colonne/signal : la colonne gagne, warning « signal shadowed by bar
   column » pour que ce soit visible.

`resolve_execution_price` prend le résolu ; le cas `Series` indexe un slice, le cas
`PerBar` est le code actuel déplacé tel quel.

### 2.2 Indexation : `signal_row`, pas `row`

La série est lue à `signal_row = row - signal_delay`, le même statut d'information que
le sizing et que les niveaux d'entrée conditionnelle (642994f : « no look-ahead the
sizing does not already have »).

Avec le défaut livré `signal_delay = 0` (convention market-on-close, parité
vectorbt, cf. eac22e6), `signal_row == row` : l'utilisateur peut composer le clip à
l'open de la barre d'exécution dans le DSL (voir 2.5) sans surprise. Avec
`signal_delay > 0`, la série est lue sur la barre de signal ; à documenter comme tel.

C'est le point où on fait MIEUX que vectorbt : leur `price=` accepte n'importe quel
tableau et laisse l'utilisateur se remplir à un prix calculé sur le close de la barre
courante, sans un mot.

### 2.3 NaN de warm-up

Fallback au close + warning, le précédent exact d'`AtVwap` (`fill.rs:67`). Compté dans
les warnings. Pas de rechute du bug b715248 ici : le prix d'exécution ne gate pas la
création d'ordre, donc aucune interaction avec le dedup de cible.

### 2.4 Ce qui ne change pas

- **Fees et slippage** : chemin market normal, taker + slippage sur le niveau. C'est
  la sémantique live du cas d'usage (ordre au marché déclenché au franchissement).
- **Sizing au close** : l'écart sizing/fill existe déjà pour `Custom` colonne ;
  `size_at_fill_price` reste réservé aux ordres d'entrée. Documenté, pas modifié.
- **`fast_path_blocker`** : refuse déjà tout non-`AtClose`, message déjà juste. Un
  sweep avec prix custom reste sur les boucles générales, et le dit.
- **Kernel rapide, CUDA** : zéro changement, zéro risque de parité FP.
- **`validate_fill_price`** : conservé tel quel ; un niveau hors de `[low, high]` de
  la barre d'exécution warn, c'est le garde-fou que vectorbt n'a pas.

### 2.5 Exemple cible (le cas utilisateur, verbatim après le chantier)

```python
# bande haute/basse en % d'une SMA horaire sur bougies fermées, diffusée au 1m
exec_level = bt.when(high >= bande_haute,
                     bt.when(open >= bande_haute, open, bande_haute),
             bt.when(low <= bande_basse,
                     bt.when(open <= bande_basse, open, bande_basse),
             close))

strat = strategie(...).signal("exec_level", exec_level)
cf.execution.execution_price = bt.ExecutionPrice.custom("exec_level")
```

Un seul mécanisme couvre l'entrée ET la sortie (le `when` choisit la bande touchée),
et le clip à l'open gère la barre qui ouvre au-delà du niveau.

---

## 3. Phases

### Phase 0 : garde-fous

- Goldens verts (`BT_UNLOCKED=1 cargo test`), baseline `runner_benchmark` consignée.
- Seuil accepté : **0 %** de régression sur le chemin `AtClose`.

### Phase 1 : moteur

- `ResolvedExecPrice` dans `fill.rs`, calqué sur `ResolvedEntryPrice`
  (`orders.rs:293`).
- Résolution par symbole avant la boucle, aux deux endroits où `resolve_entry` est
  déjà appelé (`orchestrator.rs:1666`, `:2947`) ; fusion des noms avec
  `entry_signal_names()` pour un seul appel de récolte.
- Les deux sites de consommation (`orchestrator.rs:2017`, `:3336`) passent le résolu
  et `signal_row`.
- Erreur avant la boucle sur nom inconnu.

### Phase 2 : surface, validation, observabilité

- Docstring de `ExecutionPrice.custom()` (`helpers.py:125`) : « colonne de barre ou
  signal de la stratégie », ordre de résolution, indexation, NaN.
- Warnings : shadowing colonne/signal, fallback close sur NaN (comptés).
- Message d'erreur du nom inconnu : citer les signaux disponibles.

### Phase 3 : tests

- Unitaire Rust : le fill atterrit sur la valeur de la série à `signal_row`
  (slippage compris), pas au close.
- Repro utilisateur en test Python : bandes sur SMA horaire diffusée au 1m, assert
  fill == niveau de bande sur les barres de franchissement, écart de rendement vs
  `AtClose` du bon signe.
- Parité `run()` vs lite sur le même scénario.
- Goldens inchangés quand la fonctionnalité n'est pas utilisée.
- NaN de warm-up : fallback + warning, le backtest ne meurt pas.
- `check_entry_bar` : un SL est bien déclenchable sur la barre d'entrée quand le
  fill vient d'une série.
- Nom inconnu : erreur avant la boucle, message avec les candidats.
- Shadowing : warning présent.

### Phase 4 : mesure et documentation

- Re-bench vs baseline phase 0 ; consigner dans `perf/`.
- `strategy-authoring.md` : ligne dans le tableau des modes d'exécution + section
  « fill au niveau » avec l'exemple 2.5.
- Exemple runnable dans `examples/` (le pattern bandes), comme 12fc5e6 pour les
  entrées conditionnelles.

---

## 4. Risques

| Risque | Mitigation |
|---|---|
| Régression sur le chemin sans prix custom | résolution hors boucle, `PerBar` = code actuel déplacé tel quel, goldens + bench 0 % |
| Look-ahead offert par une série mal construite | indexation `signal_row` par défaut + `validate_fill_price` qui warn hors `[low, high]` + doc explicite |
| Collision de nom colonne/signal | précédence colonne (rétro-compat) + warning |
| NaN silencieux au warm-up | fallback close compté + warning, précédent `AtVwap` |
| Utilisateur qui perd le fast path sans le savoir | `fast_path_blocker` le dit déjà ; phrase dédiée dans la doc |
| Fill flatteur (niveau jamais échangé dans la barre) | `validate_fill_price` warn ; le cas nominal (niveau entre open et high) est prouvé par la barre elle-même |

## 5. Arbitrages à trancher

1. **`Custom` élargi ou nouvelle variante `Signal(String)` ?** Recommandé : élargir
   `Custom`. Le script utilisateur marche verbatim, pas de nouvelle surface serde, la
   précédence colonne préserve l'existant. Une variante explicite reste possible plus
   tard si la collision devient un vrai problème.
2. **Indexation `signal_row` ou `row` ?** Recommandé : `signal_row`. Avec le défaut
   `signal_delay=0` c'est identique à `row` ; avec un délai, c'est le seul choix
   cohérent avec le sizing et les entrées conditionnelles.
3. **NaN : fallback close + warning, ou pas de trade ?** Recommandé : fallback +
   warning (précédent `AtVwap`). « Pas de trade » créerait une interaction avec le
   dedup de cible, exactement la classe de bug de b715248.

## 6. Hors périmètre, à traiter à part

- **Repricing des ordres au repos** (niveau `Series` gelé à la création,
  `orchestrator.rs:664`) : chantier réel mais distinct, ne débloque pas ce cas.
- **`import_dataframe` qui écrase `vwap` (recalculé ohlc4) et jette `bid`/`ask` en
  silence** : piège mesuré, mérite au minimum un warning. Indépendant de ce plan.
- **GPU / kernel rapide** : un prix custom reste sur les boucles générales ;
  n'ouvrir que si la demande le justifie, comme la phase 3b des entrées
  conditionnelles.
