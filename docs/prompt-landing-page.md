# Prompt — Landing Page backtester-engine

Prompt à utiliser avec un LLM capable de générer du code (Claude, Cursor, etc.) pour produire la landing page.

---

## PROMPT

```
Tu es un développeur frontend senior spécialisé en sites produit pour des outils quantitatifs et fintech institutionnels.

Ta mission : créer la LANDING PAGE pour "backtester-engine", une librairie Python de backtesting quantitatif avec un core Rust. UNE SEULE PAGE pour l'instant.

====================================================================
CONTEXTE PRODUIT
====================================================================

backtester-engine est une lib Python (`pip install backtester-engine`) avec un moteur Rust sous le capot. Elle cible des quant traders sérieux, semi-pros et pros, qui veulent une infrastructure de backtesting de qualité institutionnelle.

POSITIONNEMENT : Infrastructure de backtesting pour la recherche systématique.
- Les outils existants (vectorbt, backtrader, pandas maison) prennent des raccourcis sur la modélisation d'exécution : fees flat, slippage naïf ou ignoré, pas de funding rates.
- backtester-engine modélise chaque composant d'exécution indépendamment, et tourne 10-100x plus vite grâce au core Rust + Arrow columnar.
- Ce n'est PAS un outil crypto flashy. C'est de l'infrastructure de recherche quantitative qui supporte entre autres les marchés crypto.

FEATURES CLÉS :
1. Performance : Core Rust + Arrow columnar. Backtest 1 an de barres 1s en <200ms. Sweep 1000 paramètres en <30s.
2. Modélisation d'exécution : Slippage volume-impact, partial fills, funding rates perpetuels, borrow costs, maker/taker fees, limit orders, trailing stops. Chaque composant modélisé indépendamment, configurable par venue.
3. Research workflow : Walk-forward optimization, parameter stability analysis, heatmaps 2D, Monte Carlo resampling, replay déterministe bit-for-bit via manifests.
4. Expression DSL : Déclaration de signaux composable, auto-vectorisé. 60+ indicateurs built-in (SMA, EMA, RSI, MACD, Bollinger, ATR, Kalman filter, linreg, z-score, etc.)
5. Multi-asset natif : Cross-sectional ops (ranking, mean across symbols). Stat arb, pairs trading, basket strategies nativement.
6. Tearsheets : Equity curves, drawdown, monthly returns heatmap, rolling Sharpe, export pour reporting et due diligence.

EXEMPLE DE CODE — Stat Arb (OU model) :
```python
import backtester_engine as bt
from backtester_engine.expr import col, lit, symbol_ref
from backtester_engine.indicators import close, kalman

# Spread construction
pair_close = symbol_ref("ETHUSDT", "close")
ratio = close / (pair_close + lit(1e-12))

# Kalman equilibrium
equilibrium = kalman(ratio, q=1e-4, r=1e-2)
spread = ratio - equilibrium

# OU parameter (mean-reversion speed)
neg_theta = spread.linreg_slope(28)

# Z-score signal
spread_z = spread.zscore(28).ewm_mean(8)
raw = (lit(0.0) - spread_z) * lit(0.05)
signal = bt.when(
    (neg_theta < lit(0.0)) & ((spread_z > lit(0.5)) | (spread_z < lit(-0.5))),
    raw,
    lit(0.0),
)

strategy = (
    bt.Strategy.create("ou_stat_arb")
    .signal("spread_z", spread_z)
    .signal("neg_theta", neg_theta)
    .signal("signal", signal)
    .size(col("signal"))
)

result = bt.run(strategy, config, store)
print(result.summary())
```

EXEMPLE — Momentum + Sweep :
```python
import backtester_engine as bt
from backtester_engine.indicators import close, sma, rsi

fast = sma(close, 20)
slow = sma(close, 50)
signal = bt.when((fast > slow) & (rsi(close, 14) < 70), 1.0, 0.0)

strategy = (
    bt.Strategy.create("momentum")
    .signal("signal", signal)
    .size(signal * 0.25)
    .stop_loss(pct=3.0)
    .take_profit(pct=8.0)
)

sweep = bt.run_sweep(
    strategy,
    {"fast": [10, 15, 20, 30], "slow": [40, 50, 60, 80, 100]},
    config, store
)
print(sweep.best("sharpe").summary())
```

MOCK OUTPUT — result.summary() :
Ce mock doit apparaître dans le hero, à côté du code, pour montrer ce que le produit PRODUIT :
```
────────────────────────────────────────────────────────
  backtester-engine v0.9.2 — momentum / BTCUSDT 1m
  Period: 2024-01-01 → 2025-01-01 (365 days)
  Execution: 0.187s (31.5M bars processed)
────────────────────────────────────────────────────────

  Net Return        +18.42%     Sharpe Ratio     1.83
  Max Drawdown      -7.31%      Sortino Ratio    2.61
  Win Rate          58.4%       Profit Factor    1.72
  Total Trades      1,247       Avg Duration     4.2h

  Fees Paid         $2,841      Funding Costs    $1,203
  Slippage Impact   $947        Net of Costs     +14.11%

  Walk-Fwd Sharpe   1.41        Parameter Stability  0.87
  Monte Carlo p5    +6.2%       Monte Carlo p95      +22.8%
```

PRICING (3 tiers) :
- Community (gratuit) : Core engine, 5 indicateurs, 1 asset, pas de sweep, pas de walk-forward. Suffisant pour évaluer.
- Pro ($49/mo ou $399/an) : Tout. Tous les indicateurs, multi-asset, sweeps parallèles, walk-forward, Monte Carlo, tearsheets, support prioritaire.
- Team ($149/mo ou $1,199/an) : Pro x 5 seats. Shared workspace, résultats partagés.

====================================================================
AUDIENCE CIBLE
====================================================================

L'audience est TECHNIQUE et SÉRIEUSE. Pense au profil type qui utilise QuantConnect, IBKR, ou qui lit des papers sur SSRN. Pas des crypto bros.

1. QUANT TRADER SÉRIEUX (cible principale)
   - Trader systématique avec $100K-$5M de capital
   - Python-heavy, Jupyter, opère sur perps + spot
   - Valorise la rigueur, la reproductibilité, la vitesse d'itération
   - Pain : gap backtest-live à cause de mauvaise modélisation d'exécution
   - Cherche un outil qui inspire confiance dans ses résultats

2. PORTFOLIO MANAGER / GESTIONNAIRE DE STRATÉGIES
   - Gère un portefeuille de stratégies systématiques
   - Besoin de tearsheets professionnels, walk-forward validation
   - Pain : difficile de justifier les résultats backtest auprès d'allocateurs ou partenaires

3. RESEARCH TEAM (2-5 personnes)
   - Petite équipe de recherche qui partage l'infra
   - Lead technique décide de l'outil
   - Pain : scripts maison non reproductibles, lents, pas de workflow standardisé

4. QUANT STUDENT (funnel long terme)
   - Apprend le quant, construit un portfolio de stratégies
   - Free tier, convertit en Pro dans 1-3 ans

====================================================================
STRUCTURE DE LA LANDING PAGE
====================================================================

La page doit se lire comme un document technique bien structuré, pas comme une page marketing SaaS classique.

1. NAVBAR
   - Logo sobre (monogramme "bt" ou texte "backtester-engine" en mono)
   - Liens : Features, Performance, Pricing, Documentation, GitHub
   - CTA discret "Get started" (pas un bouton flashy)
   - Sobre, pas de burger menu fancy. Juste fonctionnel.

2. HERO (la section la plus importante)
   - Headline sobre et factuel : PAS de "Your backtest is lying" ou tout truc marketing. Quelque chose comme "Backtesting infrastructure for systematic strategies" — froid, institutionnel, confiant.
   - Sous-titre court : "Python library, Rust core. Realistic execution modeling, sub-second performance, reproducible research workflows."
   - CTA : `pip install backtester-engine` (copiable) + lien "View source →"
   - En dessous : TWO-PANEL layout côte à côte :
     - Panel gauche : code snippet (strategy.py) avec syntax highlighting
     - Panel droit : output terminal (result.summary()) — LE MOCK CI-DESSUS
   - C'est le hero qui doit vendre le produit. Le visiteur voit immédiatement : je code ça → j'obtiens ça.
   - PAS de badges, PAS de "Now in public beta", PAS de vanity metrics dans le hero.

3. EXECUTION MODELING COMPARISON (remplace "Why your backtest is wrong")
   - PAS de cartes avec icônes. Une TABLE DE COMPARAISON propre :
     Dimension | Typical framework | backtester-engine
   - Lignes : Fee model, Slippage, Funding rates, Borrow costs, Order types, Reproducibility
   - Sobre, factuel, comme un tableau dans un prospectus ou une doc technique.
   - Pas de "Your backtest is lying" — pas de provocation marketing.

4. FEATURES
   - Grid clean (2 ou 3 colonnes)
   - Chaque feature : titre + description concise + détails techniques en tags mono
   - Catégories : Performance, Execution realism, Research workflow, Expression DSL, Multi-asset, Outputs
   - Pas de icônes colorées, pas de badges. Sobre.

5. PERFORMANCE / BENCHMARK
   - Section deux colonnes : texte explicatif à gauche, données à droite
   - Côté données : tableau ou barres minimalistes (PAS d'animation flashy)
   - Données : backtester-engine vs vectorbt vs backtrader (temps single backtest + sweep)
   - Disclaimer benchmark discret en dessous
   - Un encadré pour le sweep : "27s — 1,000 parameter combinations"

6. RESEARCH WORKFLOW (stat arb example)
   - Deux colonnes : code (stat_arb.py) + explication
   - Montre le code stat arb complet avec le DSL
   - À droite : 3-4 blocs sobres expliquant les concepts (Expression DSL, Symbol references, Walk-forward, Manifest replay)

7. PRICING
   - 3 colonnes, style comparison matrix
   - PAS de badge "Most popular" flashy. Si highlight Pro, juste une bordure subtile.
   - Annual pricing en note discrète, pas de toggle
   - Features listées avec checkmarks simples (SVG inline) et X pour excluded
   - Sobre comme une grille tarifaire IBKR ou Bloomberg

8. FAQ
   - Accordion simple avec + qui rotate en ×
   - Questions : marchés supportés, comparaison vectorbt, besoin de Rust ?, data custom, reproducibilité, Community tier limits, notebook support, sweep parallelism
   - Réponses factuelles et concises

9. FOOTER
   - Minimal : logo, liens (Product, Resources, Company), GitHub, Discord
   - Disclaimer légal : "Past performance is not indicative of future results."

====================================================================
DESIGN & TONE — CRITIQUE
====================================================================

RÉFÉRENCES VISUELLES :
- QuantConnect (https://www.quantconnect.com) — propre, technique, pas flashy
- Interactive Brokers — dense, data-driven, fonctionnel
- Two Sigma / AQR / Man Group — institutionnel, sobre, confiance
- Bloomberg Terminal UI — densité d'information, monospace, pas de décoration
- Vercel/Linear UNIQUEMENT pour la qualité d'exécution technique (typographie, spacing, polish) — PAS pour le style marketing

CE QU'ON NE VEUT PAS :
- ❌ Esthétique "crypto" : néon vert, cyan flashy, fusées, lunes, gradient rainbow
- ❌ Style "SaaS indie" : badges colorés, "Most Popular", emojis, copy trop casual
- ❌ Style "Stripe/Linear clone" : hero gradients spectaculaires, animations wow
- ❌ Marketing copy agressif : "Your backtest is LYING", "Revolutionary", "Game-changer"
- ❌ Vanity metrics dans le hero ("10K+ strategies backtested")
- ❌ Parallax, animations complexes, effets au scroll exagérés

CE QU'ON VEUT :
- ✅ Sobre, propre, professionnel — inspire la confiance institutionnelle
- ✅ Le PRODUIT parle de lui-même (code input → data output dans le hero)
- ✅ Densité d'information maîtrisée : tableaux, données, pas de padding excessif
- ✅ Palette froide : fond sombre profond (#0b0c10), bleu institutionnel froid (#5b7ff5) comme accent discret, texte blanc cassé (#e8e9ed), gris structuré pour le muted (#8b8fa3) et dim (#555872)
- ✅ Pas de couleur d'accent omniprésente — l'accent bleu est utilisé avec parcimonie (liens, headers de colonnes, bordure active pricing)
- ✅ Typographie : Inter (texte) + JetBrains Mono (code, données). Pas de typo display fancy.
- ✅ Borders fines et discrètes (#1e2030) pour structurer, pas des cards avec ombres
- ✅ Syntax highlighting sobre (github-dark-default ou One Dark Pro)
- ✅ Animations : uniquement fade-in subtil au scroll (opacity + translate Y, 16px, 0.4s). Rien d'autre.
- ✅ Mobile responsive mais le site est pensé desktop-first (l'audience est sur desktop)
- ✅ Le site doit ressembler à un outil, pas à une publicité

STACK TECHNIQUE :
- Next.js 15 (App Router)
- Tailwind CSS v4
- Framer Motion (uniquement pour les fade-in sobres)
- Shiki pour syntax highlighting
- Déployable sur Vercel
- PAS de lucide-react ou icon library — SVG inline quand nécessaire (checkmarks, copy icon, menu)

====================================================================
CONTRAINTES
====================================================================

- LANDING PAGE UNIQUEMENT — une seule page
- Le site doit être COMPLET et fonctionnel, pas un template à moitié rempli
- Tout le contenu doit être réel (pas de "Lorem ipsum")
- Les code examples doivent utiliser la VRAIE API de backtester-engine (exemples fournis ci-dessus)
- Les prix doivent correspondre au pricing indiqué
- Pas de promesses sur les rendements ou la performance financière (compliance)
- Le CTA principal est "pip install backtester-engine" (copiable, pas un formulaire)
- Lien Discord : https://discord.gg/backtester-engine
- Lien GitHub : https://github.com/backtester-engine/backtester-engine
- Disclaimer footer : "Past performance is not indicative of future results."
- PAS de mention de hedge funds ou d'institutions comme clients existants
- Tone : on ne prétend pas être Bloomberg. On construit un outil sérieux pour des gens sérieux. La confiance vient de la rigueur technique, pas du marketing.

====================================================================
CE QUE JE VEUX EN OUTPUT
====================================================================

Le code source complet de la landing page :
- Tous les fichiers Next.js (layout.tsx, page.tsx, components/)
- globals.css avec la config Tailwind v4 (@theme)
- package.json
- tsconfig.json
- postcss.config.mjs
- next.config.ts
- Tout le contenu intégré directement (pas de CMS)
- Prêt à déployer avec `npm run build && npm run start`
- Structure de fichiers : app/ et components/ à la racine (pas de src/)
```
