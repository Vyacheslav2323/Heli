# DESIGN

## Direction
Utility-first product UI: dense enough for debugging, visually calm, and not dashboard-cluttered.

## Composition Rules
- Single full-bleed visual plane for GLB scene.
- Overlay layer only for masks and minimal status cues.
- One bottom chat dock with text + audio controls.

## Interaction Rules
- Left click for segmentation point prompt.
- Camera orbit remains available with non-conflicting gesture.
- Failures surface as non-blocking but persistent alerts.

## Accessibility Baseline
- Sufficient contrast for overlays and controls.
- Keyboard focus visible for chat controls.
- Touch target minimum 40px for primary toggles.
