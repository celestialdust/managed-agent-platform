# The README's diagrams

Eight diagrams, each kept twice: a `.mmd` Mermaid source, which is what you edit, and a
rendered `.svg`, which is what `README.md` links to. The numbering starts at 01 and skips
nothing that is here; two more once existed for a section the README no longer has, and
they went out with it rather than sitting unreferenced.

**Why not a ```mermaid fence, which GitHub renders on its own and needs none of this.**
It renders the diagram inside a viewer with a pan-and-zoom control panel pinned to the
diagram's right edge. The panel is always visible rather than shown on hover, and it sits
*over* the drawing — measured on the published page, it covered a node's label outright.
There is no way to style or suppress it from Markdown, because it is GitHub's own chrome
around the fence rather than anything the fence contains. An `<img>` gets none of it.

Two consequences worth knowing before you edit one:

- **The SVGs use text labels, not HTML labels.** A browser does not render `foreignObject`
  inside an image, and an HTML label is exactly that — so an HTML-label render would
  publish ten blank rectangles. `htmlLabels: false` is not a style preference here.
  Labels use Mermaid's markdown strings (backtick-quoted, `**bold**`, real newlines),
  which survive the switch; `<b>` and `<br/>` do not.
- **They carry an explicit white background** and are not theme-aware. On GitHub's dark
  theme they read as a light figure on a dark page, which is legible and deliberate.
  Making them follow the theme means a second render per diagram and a `<picture>` element
  per figure — worth doing if it ever bothers anyone, not done today.

## Regenerating

Any Mermaid renderer works, as long as it is told not to use HTML labels. With
[mermaid-cli](https://github.com/mermaid-js/mermaid-cli):

```sh
npx -y @mermaid-js/mermaid-cli -i 05-topology.mmd -o 05-topology.svg \
  --configFile config.json --backgroundColor '#ffffff'
```

`config.json` next to this file holds the settings the committed SVGs were rendered with.
Render every diagram through the same config, or one figure comes out in a different
palette from the nine beside it.

**Both files are the same fact, so change them together.** An edited `.mmd` that was never
re-rendered leaves `README.md` showing the old drawing while the source says otherwise, and
nothing here checks — the two are kept in step by whoever edits them.
