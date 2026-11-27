#set page(width: auto, height: auto, margin: 0pt)
#set image(width: 100%)

#let hdr(label) = align(center + horizon)[#strong(label)]

#grid(
  // columns: (auto, ..(4cm,) * 5),
  columns: ((10cm,) * 5),
  gutter: 3pt,
  image("plots/2a_q_gpt4/r1_vs_r2.pdf"), [], [],
  grid.cell(colspan: 2, rowspan: 2, align: center + bottom, scale(150%, reflow: true, image(
    "plots/keys/key_ques.pdf",
    width: auto,
  ))),

  image("plots/2a_q_gpt4/r1_vs_r3.pdf"), image("plots/2a_q_gpt4/r2_vs_r3.pdf"), [],

  image("plots/2a_q_gpt4/r1_vs_r4.pdf"),
  image("plots/2a_q_gpt4/r2_vs_r4.pdf"),
  image("plots/2a_q_gpt4/r3_vs_r4.pdf"),
  [],
  [],

  image("plots/2a_q_gpt4/r1_vs_r5.pdf"),
  image("plots/2a_q_gpt4/r2_vs_r5.pdf"),
  image("plots/2a_q_gpt4/r3_vs_r5.pdf"),
  image("plots/2a_q_gpt4/r4_vs_r5.pdf"),
  [],

  image("plots/2a_q_gpt4/r1_vs_r6.pdf"),
  image("plots/2a_q_gpt4/r2_vs_r6.pdf"),
  image("plots/2a_q_gpt4/r3_vs_r6.pdf"),
  image("plots/2a_q_gpt4/r4_vs_r6.pdf"),
  image("plots/2a_q_gpt4/r5_vs_r6.pdf"),
)
