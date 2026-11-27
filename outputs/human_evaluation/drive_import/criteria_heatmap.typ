#set page(width: auto, height: auto, margin: 10pt)
#set image(width: 100%)

#let hdr(label) = align(center + horizon)[#strong(label)]

#grid(
  // columns: (auto, ..(4cm,) * 5),
  columns: ((10cm,) * 3),
  gutter: 3pt,
  hdr("Both GPT models"), hdr("GPT-4o"), hdr("GPT-5.2"),
  image("plots/criteria_heatmaps/criteria_correlation_gpt.pdf"),
  image("plots/criteria_heatmaps/criteria_correlation_4o.pdf"),
  image("plots/criteria_heatmaps/criteria_correlation_5.2.pdf"),
)
