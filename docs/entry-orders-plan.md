# Plan — ordres d'entrée conditionnels (market / limit / stop à prix déterminé)

Branche : `claude/limit-market-order-entries-mqvww2` (depuis `main` @ `c483032`, 0.14.1).

## État d'avancement

| Phase | Statut |
|---|---|
| 0 — garde-fous perf | fait (baseline benchée, goldens verts) |
| 1 — modèle de prix d'entrée (boucles générales) | **fait** |
| 2 — surface utilisateur, validation, observabilité | **fait** |
| 3 — ordre au repos dans le kernel rapide + CUDA | **pas fait** — voir §3 |
| 3b — prix piloté par série sur GPU | pas fait |
| 4 — mesure | fait — `perf/entry-orders.md` |

Ce qui est livré couvre les quatre déclencheurs (`Limit`, `Stop`, `StopLimit`,
`MarketIfTouched`), les trois formes de prix (`OffsetBps`, `Absolute`, `Signal`), le
time-in-force existant, le sizing optionnel au prix de l'ordre, la validation à la
compilation et le décompte des ordres non remplis. Une entrée conditionnelle reste hors du
kernel rapide : `fast_path_blocker` continue de la refuser et de le dire.

But : permettre à l'utilisateur de décrire **où** son entrée se remplit — pas seulement
« au close de la barre suivante » — sans casser la vitesse ni les résultats existants.

Contrainte cardinale : **zéro coût quand la fonctionnalité n'est pas utilisée**, et pas de
quatrième transcription de la règle de fill (cf. `refactor(engine): one fill/sizing rule
instead of three transcriptions`).

---

## 1. État des lieux

### Ce qui existe

`OrderConfig.limit_entry` (`crates/bt-core/src/orders.rs:27`) :

```rust
pub struct LimitOrderConfig {
    pub offset_bps: f64,            // distance depuis le close de la barre de SIGNAL
    pub time_in_force: TimeInForce, // GTC | GTB(n) | IOC
}
```

Réellement branché, sur les deux boucles générales :

| Élément | Emplacement |
|---|---|
| Calcul du prix limite (close du signal, pas de look-ahead) | `orchestrator.rs:1628-1638`, `:2961-2969` |
| Gate de fill (`low <= limit` achat, `high >= limit` vente) | `orchestrator.rs:1679-1704` |
| TIF (IOC annule, GTB(n) expire, GTC persiste) | `orchestrator.rs:1689-1701` |
| Fill au prix limite, frais **maker**, slippage nul | `orchestrator.rs:1751-1762` |
| État de l'ordre au repos | `PendingOrder`, `orchestrator.rs:442-462` |
| Bracket SL/TP armé dès la première fraction remplie | `orchestrator.rs:1791-1800` |
| Tests | `crates/bt-core/tests/backtest_orders.rs:601`, `:647`, `:687` |

Exposition : dict brut via `ExecutionConfig.orders` (`config.py:24`) ou champ `orders` du
StrategyDef (`compiler.rs:24`), résolu par `effective_orders` (`orchestrator.rs:4578`) —
le per-stratégie écrase le global.

### Ce qui manque

1. **Aucun prix absolu ni piloté par une expression.** Seul « X bps sous le close du signal »
   est exprimable. Pas de « achète à 60 000 », pas de « achète sur l'EMA20 / le VWAP /
   `close - 2*atr(14)` / le low de la veille ».
2. **Pas de stop entry (breakout).** Le sens est codé en dur en passif
   (`orchestrator.rs:1682-1686`).
3. **Pas de gestion de gap à l'entrée.** Le fill se fait au prix limite exact, sans regarder
   l'open. Pour un *limit* c'est conservateur (on paye la limite alors que le gap aurait donné
   mieux) — acceptable. Pour un *stop* ce serait faux et **optimiste** : la règle gap-aware
   existe déjà côté sortie (`sim_fast_lite_core_single`, `orchestrator.rs:5931-5946`) et doit
   être réutilisée telle quelle.
4. **Pas de méthode dans le builder Python.** `Strategy` a `.stop_loss()`, `.take_profit()`,
   `.trailing_stop()` (`strategy.py:124-160`) mais rien côté entrée.
5. **`ExitReason::LimitExpiry` déclaré et jamais émis** (`orders.rs:124`) : un ordre annulé
   disparaît sans trace. L'utilisateur ne peut pas savoir que sa stratégie n'a jamais été
   remplie.
6. **Le sizing ignore le prix de fill.** `sanitize_target` reçoit le close de la barre courante
   (`orchestrator.rs:1618-1625`) : en `FractionOfEquity`, une entrée limite à 2 % sous le close
   achète 2 % de notionnel de trop.

### Le vrai coût : le fast path

`fast_path_blocker` refuse **toute** limit entry, sans condition :

```rust
// orchestrator.rs:4927
if orders.limit_entry.is_some() {
    return Some("the strategy has a conditional entry order (limit / stop)");
}
```

Commentaire à l'appui (`orchestrator.rs:4612`) : *« A LIMIT ENTRY blocks regardless: it can rest
unfilled across bars, which the single-bar fill assumption here cannot express. »*

Conséquence : une entrée conditionnelle sort du kernel rapide **et** du GPU. Sur un sweep, c'est
le facteur ~25× qui disparaît. Si on rend la fonctionnalité attrayante sans toucher à ça, les
utilisateurs vont la mettre partout et perdre la vitesse qui est l'argument du produit.

Les trois niveaux d'exécution à garder en tête :

| Niveau | Fonction | Ordres au repos ? |
|---|---|---|
| Boucle générale (`run`, `run_lite_on_aligned`) | `orchestrator.rs:1357`, `:2154` | oui |
| Kernels rapides CPU | `simulate_fast:4995`, `simulate_fast_lite:5225`, `sim_fast_lite_core_single:5869` | non |
| Kernel CUDA (transpilé 1:1 du précédent) | `gpu_sweep.rs:1115` (`bracket_src`) | non |

---

## 2. Les possibilités (espace de conception)

### Axe A — déclencheur

| Mode | Sémantique | Frais | Statut |
|---|---|---|---|
| `Market` | fill à la barre d'exécution selon `execution_price` | taker + slippage | existe (défaut) |
| `Limit` | fill si le prix **touche** le niveau (passif) | maker, pas de slippage | existe |
| `Stop` | fill si le prix **franchit** le niveau (breakout) | taker + slippage + gap | **à faire** |
| `StopLimit` | déclenché au stop, rempli au mieux à la limite | maker | optionnel, phase ultérieure |

`Stop` est le symétrique exact du `check_stop` de sortie ; le code de trigger et de gap existe
déjà, il n'y a qu'à l'appeler côté entrée.

### Axe B — ancrage du prix

| Variante | Exemple | Coût runtime |
|---|---|---|
| `OffsetBps(f64)` | `-25 bps sous le close du signal` | nul (existe) |
| `Absolute(f64)` | `60_000.0` | nul (scalaire) |
| `Signal(String)` | `"entry_px"` avec `entry_px: "ema(close,20)"` | une slice `&[f64]` de plus |

`Signal` couvre tout le reste (ATR, VWAP, plus-haut de N barres, niveau de Fibonacci, prix
externe injecté en colonne exo) sans inventer de mini-langage : la série est déjà matérialisée
dans `symbol_envs` au moment où le sizing est évalué (`orchestrator.rs:1140-1175`), il suffit de
l'extraire comme on extrait `position_sizing`, de la passer dans `expand_to_fine_resolution` et
de la slicer par symbole comme `target_slices` (`orchestrator.rs:1314`).

`Absolute` est redondant avec `Signal` (`entry_px: "60000"`) mais reste utile : c'est un
scalaire, donc il ne bloque pas le GPU et ne coûte pas une série.

### Axe C — durée de vie

`GTC` / `GTB(n)` / `IOC` existent et suffisent. À ajouter seulement :
- annulation quand le signal s'inverse (aujourd'hui l'override existe mais n'est pas explicite,
  `orchestrator.rs:1575-1594`) ;
- **observabilité** de l'annulation (voir §3 phase 2).

### Axe D — sizing

Deux comportements, à rendre explicite :
- `size_at_signal_close` (actuel, rétro-compatible) ;
- `size_at_fill_price` : la quantité est calculée sur le prix limite, ce que l'utilisateur
  attend quasi toujours.

Défaut proposé : garder l'actuel pour ne rien casser, exposer l'option, la documenter.

---

## 3. Plan par phases

### Phase 0 — garde-fous (avant toute ligne de feature)

- `cargo bench -p bt-core` : figer `single_asset_50k`, `sweep_1000`, `multi_asset_10sym`
  (méthodo min-médiane sur 3 passes, cf. `perf/BASELINE.md`).
- Test de non-régression « zéro coût » : une stratégie sans `orders` doit produire un résultat
  **bit-à-bit** identique avant/après. Les tests de parité existent déjà
  (`test_fast_lite_core_matches_simulate_fast_lite`, `orchestrator.rs:6844`) — les étendre.

### Phase 1 — modèle de prix d'entrée (boucle générale, CPU)

```rust
pub enum EntryPrice {
    OffsetBps(f64),
    Absolute(f64),
    Signal(String),
}

pub enum EntryTrigger { Market, Limit, Stop }

pub struct EntryOrderConfig {
    pub price: EntryPrice,
    pub trigger: EntryTrigger,
    pub time_in_force: TimeInForce,
    pub size_at_fill_price: bool,
}
```

- Rétro-compatibilité serde : `{"offset_bps": 10.0, "time_in_force": "GTC"}` continue de
  désérialiser en `Limit` + `OffsetBps` (alias serde + `#[serde(default)]`).
- Résolution du prix : une seule fonction `resolve_entry_price(...) -> f64`, appelée depuis les
  **deux** boucles générales — pas deux transcriptions.
- Trigger `Stop` : réutiliser la règle gap-aware de la sortie, sans la recopier.
- Validation : `Signal(name)` inconnu → erreur dans `validate_strategy`
  (`crates/bt-strategy/src/validate.rs`), pas un NaN silencieux à l'exécution.
- Tests : fill, non-fill, expiration, gap, short, multi-bar fill, `Signal` piloté par indicateur.

### Phase 2 — surface utilisateur

- `Strategy.limit_entry(...)`, `.stop_entry(...)`, `.market_entry(...)` (`strategy.py`), sur le
  modèle exact de `.stop_loss()`.
- `OrderConfig` typé côté Python plutôt que dict brut (`config.py:10-56`).
- Émission de `ExitReason::LimitExpiry` : une entrée annulée doit être visible. Comme elle ne
  produit pas de trade, l'exposer via un compteur (`orders_cancelled`) dans le résultat + un
  warning quand le taux de non-remplissage dépasse un seuil — un backtest « parfait » qui n'a
  jamais rempli est le piège n°1 des entrées limites.
- Doc : section dédiée dans `docs/strategy-authoring.md`, plus un exemple dans `examples/`.

### Phase 3 — perf : l'ordre au repos dans le kernel rapide

C'est la phase qui protège la vitesse. Même patron que `LiteBracket` (`orchestrator.rs:5751`) :
struct plate `Copy`, `NaN` = absent, pas d'enum, pas de branche quand la feature est absente.

```rust
#[derive(Clone, Copy, Default)]
struct LitePending {
    limit_price: f64,   // NaN = pas d'ordre au repos
    remaining: f64,
    bars_alive: f64,
    is_buy: f64,        // encodé en f64 pour rester transpilable
}
```

- Ajouter au `sim_fast_lite_core_single` en `Option<...>` comme `exits`, pour que le chemin
  sans ordre reste byte-for-byte identique.
- Élargir `fast_path_blocker` : accepter `Limit`/`Stop` quand le prix est `OffsetBps` ou
  `Absolute`. Garder le blocage pour `Signal` **tant que** la série n'est pas uploadée
  (phase 3b), et surtout garder le message explicite — c'est ce qui dit à l'utilisateur ce
  qu'il paye.
- CUDA : ajouter un `pending_src` sur le modèle de `bracket_src` (`gpu_sweep.rs:1115`) et étendre
  `pack_cfg` (`gpu_sweep.rs:1084`). Ordre des opérations FP **strictement** préservé, sinon la
  parité bit-à-bit saute.
- Tests de parité CPU/GPU étendus aux ordres au repos.

### Phase 3b (optionnelle) — prix d'entrée piloté par série sur GPU

Uploader la série `Signal(name)` comme une colonne de plus, au même titre que les closes. Ne le
faire que si la mesure de la phase 4 montre que `Signal` est le cas d'usage dominant.

### Phase 4 — mesure et documentation

- Re-bench, comparaison à la baseline de la phase 0, seuil de régression accepté : **0 %** sur
  le chemin sans ordres, à documenter sur le chemin avec ordres.
- Consigner dans `perf/` comme les campagnes précédentes.

---

## 4. Risques

| Risque | Mitigation |
|---|---|
| Divergence FP CPU/CUDA (kernel transpilé à la main) | une seule règle partagée, ordre des opérations figé, tests de parité étendus |
| Régression silencieuse sur le chemin sans ordres | golden bit-à-bit en phase 0, `Option` partout |
| Utilisateur qui perd le fast path sans le savoir | message de `fast_path_blocker` explicite, déjà surfacé par le sweep GPU |
| Backtest flatteur parce que rien n'a été rempli | compteur d'annulations + warning (phase 2) |
| Sizing incohérent avec le prix de fill | option explicite, défaut inchangé |

## 5. Arbitrages à trancher

1. **Stop entry (breakout) dans le lot initial ?** Le code de trigger et de gap existe côté
   sortie, le coût marginal est faible — recommandé oui.
2. **Phase 3 (kernel rapide + CUDA) maintenant ou après retour utilisateur ?** C'est la moitié
   de l'effort. Sans elle, la fonctionnalité marche mais coûte le fast path.
3. **`size_at_fill_price` par défaut ?** Plus juste, mais change les résultats des stratégies
   existantes utilisant `limit_entry`.
