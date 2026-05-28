---
name: AI Creative Insights System
colors:
  surface: '#081425'
  surface-dim: '#081425'
  surface-bright: '#2f3a4c'
  surface-container-lowest: '#040e1f'
  surface-container-low: '#111c2d'
  surface-container: '#152031'
  surface-container-high: '#1f2a3c'
  surface-container-highest: '#2a3548'
  on-surface: '#d8e3fb'
  on-surface-variant: '#cbc3d7'
  inverse-surface: '#d8e3fb'
  inverse-on-surface: '#263143'
  outline: '#958ea0'
  outline-variant: '#494454'
  surface-tint: '#d0bcff'
  primary: '#d0bcff'
  on-primary: '#3c0091'
  primary-container: '#a078ff'
  on-primary-container: '#340080'
  inverse-primary: '#6d3bd7'
  secondary: '#bec6e0'
  on-secondary: '#283044'
  secondary-container: '#3f465c'
  on-secondary-container: '#adb4ce'
  tertiary: '#7bd0ff'
  on-tertiary: '#00354a'
  tertiary-container: '#009bd1'
  on-tertiary-container: '#002d40'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e9ddff'
  primary-fixed-dim: '#d0bcff'
  on-primary-fixed: '#23005c'
  on-primary-fixed-variant: '#5516be'
  secondary-fixed: '#dae2fd'
  secondary-fixed-dim: '#bec6e0'
  on-secondary-fixed: '#131b2e'
  on-secondary-fixed-variant: '#3f465c'
  tertiary-fixed: '#c4e7ff'
  tertiary-fixed-dim: '#7bd0ff'
  on-tertiary-fixed: '#001e2c'
  on-tertiary-fixed-variant: '#004c69'
  background: '#081425'
  on-background: '#d8e3fb'
  surface-variant: '#2a3548'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  display-sm:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  data-lg:
    fontFamily: Inter
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 32px
    letterSpacing: -0.01em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  xs: 4px
  sm: 8px
  md: 16px
  lg: 24px
  xl: 32px
  container-padding: 20px
  gutter: 12px
---

## Brand & Style

The design system is engineered for precision, authority, and high-tech sophistication. It caters to a professional audience of creative directors and market analysts who require immediate, data-driven insights to evaluate advertising performance. 

The aesthetic identity is a fusion of **Corporate Modernism** and subtle **Glassmorphism**. It prioritizes clarity and analytical depth through a "dark mode first" approach, ensuring that vibrant data visualizations and video creative content remain the focal point. The interface should feel like an advanced command center: calm under pressure, impeccably organized, and intellectually sharp. 

Key visual principles include:
- **Optical Hierarchy:** Critical scores and AI-generated insights are given maximum visual prominence.
- **Controlled Vibrancy:** Intense accent colors are reserved strictly for actions and status indicators to avoid cognitive overload.
- **Technical Transparency:** Layers and blurs are used to indicate depth and context without obscuring underlying data.

## Colors

This design system utilizes a sophisticated dark-themed palette to reduce eye strain during deep data analysis and to provide a high-contrast canvas for creative assets.

- **Primary (Electric Violet):** Used for primary calls to action, active states, and AI-driven highlights. It signifies intelligence and energy.
- **Surface & Backgrounds:** A range of deep slates and charcoals (`#020617` to `#1E293B`) create a layered environment. The deepest navy is reserved for the background, while lighter slates define cards and interactive containers.
- **Semantic Scoring:** A strict three-tier color system is used for creativity evaluation:
    - **Success Green:** Scores >80.
    - **Warning Orange:** Scores 60-80.
    - **Critical Red:** Scores <60.
- **Data Accents:** Tertiary cyan is used for secondary data points or "soft" information that doesn't require immediate action but aids in context.

## Typography

The design system exclusively employs **Inter** for its exceptional legibility and systematic feel. The typographic scale is optimized for high-density information environments.

- **Data Emphasis:** Numerical values and scores use `data-lg` or `display` roles to ensure immediate recognition. 
- **Labels:** Small caps or bolded `label-md` roles with slight letter spacing are used for metadata and axis titles in charts to differentiate them from body text.
- **Hierarchy:** Use weight over size to establish hierarchy in dense views. Secondary information should utilize a mid-range gray rather than a smaller font size to maintain accessibility.

## Layout & Spacing

The layout follows a **fluid grid** model tailored for mobile efficiency, built on an 8px base unit.

- **Margins:** A standard 20px horizontal margin ensures content is comfortably inset from screen edges.
- **Grid:** Use a 4-column layout for mobile. For data tables and dashboards, content spans are fluid, while gutters are fixed at 12px to maximize horizontal space for charts.
- **Density:** High-density layouts are encouraged for evaluation screens. Use `md` (16px) spacing for primary element grouping and `sm` (8px) for internal element relationships (e.g., an icon next to its label).

## Elevation & Depth

Visual hierarchy is established through **Tonal Layers** and **Glassmorphism**, moving away from traditional heavy shadows which can muddy dark UI.

- **Surfaces:** Use progressive lightening of the background color to indicate elevation. Higher elevation elements (like popovers) use a lighter slate than the base background.
- **Glassmorphism:** Overlays, navigation bars, and floating action panels use a 20px backdrop blur with a 10% opacity white tint. This maintains context while providing a clean surface for interaction.
- **Borders:** Instead of shadows, use subtle "inner glows" or 1px strokes (`#FFFFFF` at 10% opacity) to define the edges of cards and containers against the dark background.
- **Data Pop:** High-priority scores and AI insights may feature a very soft, diffused glow in the Primary color to draw the user's eye.

## Shapes

The shape language balances professional structure with modern approachability.

- **Standard Containers:** Cards, input fields, and primary buttons use a 0.5rem (8px) or 1rem (16px) radius. 
- **Buttons:** Large action buttons use the `rounded-lg` (1rem) setting to feel distinct and tactile.
- **Data Points:** Markers in radar charts and line graphs should be circular to contrast against the geometric grid.
- **Status Pills:** Use pill-shaped (full radius) containers for status tags (e.g., "High Impact") to distinguish them from interactive buttons.

## Components

### Buttons
- **Primary:** Solid Electric Violet with white text. 16px corner radius. High-gloss finish.
- **Secondary:** Transparent background with a 1px border of the Primary color or a glassmorphic fill.
- **Icon Buttons:** Circular glassmorphic containers for utility actions (e.g., "Share", "Close").

### Data Visualization
- **Radar Charts:** Semi-transparent violet fill with a high-contrast white stroke. Grid lines in subtle gray. Points highlighted in semantic colors based on score.
- **Progress Bars:** Dual-track bars. The track is dark charcoal; the fill is the semantic status color. Use a glow effect for the "fill" edge.
- **Score Cards:** Large numerical display using `display-sm` typography. Include a small trend indicator (up/down arrow) and a brief AI-generated summary text.

### Inputs & Selection
- **Input Fields:** Deep charcoal background with a 1px bottom border that highlights in Electric Violet when focused.
- **Checkboxes/Radios:** Custom geometric designs. Active states use a solid Electric Violet fill with a white check/dot.

### Navigation
- **Bottom Bar:** Glassmorphic blur (20px) with subtle top border. Active icons utilize the Primary color and a small under-glow.

### Evaluation Cards
- High-density cards featuring a video thumbnail on the left/top and a "Scoring Cluster" on the right. Use subtle 1px borders to separate different creative versions being compared.