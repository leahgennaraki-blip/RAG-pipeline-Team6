# Repository integration: resolve paths independently of the working directory.
# Original source is preserved in archive/drive_import/original_scripts/.
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this script with Rscript, as documented in README.md")
script_path <- normalizePath(sub("^--file=", "", script_arg))
repo_root <- normalizePath(file.path(dirname(script_path), "../../.."))
data_dir <- file.path(repo_root, "data/qualitative/anonymised_ratings")
output_dir <- file.path(repo_root, "outputs/human_evaluation/reruns")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Rater correlation

# Import Packages
library(tidyverse)
library(ggpubr)
library(ggtext)

# Import Data
files <- list.files(data_dir, pattern = "*.csv$", full.names = TRUE)

# Make dataframes
## Merge all CSVs into one
all <- lapply(files, function(f) {
  read.csv(f) |>
    mutate(Rater = tools::file_path_sans_ext(basename(f)))
}) |>
  bind_rows() |>
  relocate(Rater) |>
  ## Mkae unified row for criteria
  pivot_longer(
    cols = -c(Rater, Question, Model),
    names_to = "Criteria",
    values_to = "Rating"
  )

all$Rating <- as.numeric(all$Rating)

## Make separate rows for raters
all <- all |> pivot_wider(
  names_from = "Rater",
  values_from = "Rating"
)

if (interactive()) view(all)
summary(all)

## Mean ratings across questions per model-criteria pair
all_by_unique_model_criteria <- all |>
  group_by(Model, Criteria) |>
  summarise(across(r1:r6, mean, na.rm = TRUE), .groups = "drop")

## Mean ratings across criteria per model-question pair
all_by_unique_model_questions <- all |>
  group_by(Model, Question) |>
  summarise(across(r1:r6, mean, na.rm = TRUE), .groups = "drop")

all_by_unique_model_criteria
all_by_unique_model_questions

# Pearsons r and RMSE summary table
compute_pair_stats <- function(data, rater_cols, id_cols) {
  d <- data |>
    pivot_longer(cols = all_of(rater_cols), names_to = "Rater", values_to = "Rating") |>
    filter(!is.na(Rating))

  pairs <- combn(unique(na.omit(d$Rater)), 2, simplify = FALSE)

  lapply(pairs, function(pair) {
    d |>
      filter(Rater %in% pair) |>
      pivot_wider(names_from = Rater, values_from = Rating, id_cols = all_of(id_cols)) |>
      rename(x = all_of(pair[1]), y = all_of(pair[2])) |>
      summarise(
        Rater1 = pair[1],
        Rater2 = pair[2],
        r      = cor(x, y, use = "complete.obs"),
        rmse   = sqrt(mean((x - y)^2, na.rm = TRUE))
      )
  }) |> bind_rows()
}

stats_qxc_gpt4 <- compute_pair_stats(
  all |> filter(Model == "GPT4"), c("r1", "r2", "r3", "r4", "r5", "r6"), c("Question", "Criteria")
)
stats_crit_gpt4 <- compute_pair_stats(
  all_by_unique_model_criteria |> filter(Model == "GPT4"), c("r1", "r2", "r3", "r4", "r5", "r6"), "Criteria"
)
stats_ques_gpt4 <- compute_pair_stats(
  all_by_unique_model_questions |> filter(Model == "GPT4"), c("r1", "r2", "r3", "r4", "r5", "r6"), "Question"
)

# same for GPT5
stats_qxc_gpt5 <- compute_pair_stats(all |> filter(Model == "GPT5"), c("r1", "r2", "r3", "r4", "r5", "r6"), c("Question", "Criteria"))
stats_crit_gpt5 <- compute_pair_stats(all_by_unique_model_criteria |> filter(Model == "GPT5"), c("r1", "r2", "r3", "r4", "r5", "r6"), "Criteria")
stats_ques_gpt5 <- compute_pair_stats(all_by_unique_model_questions |> filter(Model == "GPT5"), c("r1", "r2", "r3", "r4", "r5", "r6"), "Question")

summary_table <- tibble(
  Level = rep(c("Question × Criteria", "Criteria", "Question"), 2),
  Model = rep(c("GPT4", "GPT5"), each = 3),
  Median_r = c(
    median(stats_qxc_gpt4$r),  median(stats_crit_gpt4$r),  median(stats_ques_gpt4$r),
    median(stats_qxc_gpt5$r),  median(stats_crit_gpt5$r),  median(stats_ques_gpt5$r)
  ),
  Median_RMSE = c(
    median(stats_qxc_gpt4$rmse), median(stats_crit_gpt4$rmse), median(stats_ques_gpt4$rmse),
    median(stats_qxc_gpt5$rmse), median(stats_crit_gpt5$rmse), median(stats_ques_gpt5$rmse)
  )
) |> mutate(across(where(is.numeric), ~ round(., 3)))

print(summary_table)

# Plots
## (0) Criteria x Question
##    (a) GPT4 , (b) GPT5
## (1) Criteria level
##    (a) GPT4 , (b) GPT5
## (2) Question level
##    (a) GPT4 , (b) GPT5

# ── Shared aesthetics ─────────────────────────────────────────────────────────

criteria_colors <- c(
  Ambiguity      = "#e41a1c",
  Factuality     = "#377eb8",
  Generalization = "#4daf4a",
  Irrelevance    = "#984ea3",
  Narrativity    = "#ff7f00",
  Recall         = "#a65628",
  Reduction      = "#f781bf",
  Repetetion     = "#999999"
)

criteria_shapes <- c(
  Ambiguity      = 16,
  Factuality     = 17,
  Generalization = 15,
  Irrelevance    = 18,
  Narrativity    = 8,
  Recall         = 10,
  Reduction      = 7,
  Repetetion     = 4
)

# ── Build pair data ────────────────────────────────────────────────────

make_pairs <- function(d, id_cols) {
  raters <- unique(na.omit(d$Rater))
  pairs <- combn(raters, 2, simplify = FALSE)

  pair_data <- lapply(pairs, function(pair) {
    d |>
      filter(Rater %in% pair) |>
      pivot_wider(names_from = Rater, values_from = Rating) |>
      rename(x = all_of(pair[1]), y = all_of(pair[2])) |>
      mutate(Rater1 = pair[1], Rater2 = pair[2])
  }) |> bind_rows()

  pair_stats <- pair_data |>
    group_by(Rater1, Rater2) |>
    summarise(
      r = round(cor(x, y, use = "complete.obs"), 3),
      rmse = round(sqrt(mean((x - y)^2, na.rm = TRUE)), 3),
      .groups = "drop"
    )

  pair_data |> left_join(pair_stats, by = c("Rater1", "Rater2"))
}

# ── Save pair plots ──────────────────────────────────────────────────────────

save_pairs <- function(pair_data, folder, plot_fn, width = 8) {
  dir.create(file.path(output_dir, folder), showWarnings = FALSE, recursive = TRUE)
  pairs <- pair_data |> distinct(Rater1, Rater2)

  for (i in seq_len(nrow(pairs))) {
    ra <- pairs$Rater1[i]
    rb <- pairs$Rater2[i]

    plot_df <- pair_data |> filter(Rater1 == ra, Rater2 == rb)
    if (nrow(plot_df) == 0) next

    stats <- plot_df |>
      slice(1) |>
      select(r, rmse)
    subtitle <- bquote(
      italic(r) ~ "=" ~ .(sprintf("%.3f", stats$r)) * ", RMSE =" ~ .(sprintf("%.3f", stats$rmse))
    )

    p <- plot_fn(plot_df, ra, rb, subtitle)

    # sort so lower rater number always comes first in filename
    ra_sorted <- sort(c(ra, rb))[1]
    rb_sorted <- sort(c(ra, rb))[2]

    ggsave(
      filename = file.path(output_dir, folder, paste0(ra_sorted, "_vs_", rb_sorted, ".pdf")),
      plot = p,
      device = "pdf",
      width = width, height = 9, units = "cm"
    )
  }
}

# ── Plot functions ─────────────────────────────────────────────────────────────

pair_theme <- theme(
  aspect.ratio    = 1,
  plot.title      = element_text(hjust = 0.5, size = 16, margin = margin(b = 6)),
  axis.title      = element_text(size = 16, face = "bold"),
  axis.text       = element_text(size = 13),
  legend.position = "none"
)

plot_qxc <- function(plot_df, ra, rb, subtitle) {
  ggplot(plot_df, aes(x = x, y = y, color = Q_num, shape = Criteria)) +
    geom_abline(slope = 1, intercept = 0, color = alpha("#444444", 0.3), linewidth = 0.6, linetype = "dashed") +
    geom_smooth(
      inherit.aes = FALSE, aes(x = x, y = y),
      method = "lm", se = FALSE, linewidth = 0.6, color = "#444444"
    ) +
    geom_jitter(size = 1.2, width = 0.2, height = 0.2, alpha = 0.85) +
    scale_color_viridis_c(option = "plasma", name = "Question", breaks = seq(1, 24, by = 4)) +
    scale_shape_manual(values = criteria_shapes, name = "Criteria") +
    scale_x_continuous(limits = c(0.5, 5.5), breaks = 1:5) +
    scale_y_continuous(limits = c(0.5, 5.5), breaks = 1:5) +
    labs(title = subtitle, x = ra, y = rb) +
    theme_pubr(base_size = 11) +
    pair_theme
}

plot_criteria <- function(plot_df, ra, rb, subtitle) {
  ggplot(plot_df, aes(x = x, y = y, color = Criteria, shape = Criteria)) +
    geom_abline(slope = 1, intercept = 0, color = alpha("#444444", 0.3), linewidth = 0.6, linetype = "dashed") +
    geom_smooth(
      inherit.aes = FALSE, aes(x = x, y = y),
      method = "lm", se = FALSE, linewidth = 0.6, color = "#444444"
    ) +
    geom_point(size = 2) +
    scale_color_manual(values = criteria_colors, name = "Criteria") +
    scale_shape_manual(values = criteria_shapes, name = "Criteria") +
    scale_x_continuous(limits = c(1.5, 5.5), breaks = 2:5) +
    scale_y_continuous(limits = c(1.5, 5.5), breaks = 2:5) +
    labs(title = subtitle, x = ra, y = rb) +
    theme_pubr(base_size = 11) +
    pair_theme
}

plot_questions <- function(plot_df, ra, rb, subtitle) {
  ggplot(plot_df, aes(x = x, y = y, color = Q_num)) +
    geom_abline(slope = 1, intercept = 0, color = alpha("#444444", 0.3), linewidth = 0.6, linetype = "dashed") +
    geom_smooth(method = "lm", se = FALSE, linewidth = 0.6, color = "#444444") +
    geom_point(size = 2) +
    scale_color_viridis_c(option = "plasma", name = "Question", breaks = seq(1, 24, by = 4)) +
    scale_x_continuous(limits = c(1.5, 5.5), breaks = 2:5) +
    scale_y_continuous(limits = c(1.5, 5.5), breaks = 2:5) +
    labs(title = subtitle, x = ra, y = rb) +
    theme_pubr(base_size = 11) +
    pair_theme
}

# ── (0a) Q x C — GPT4 ─────────────────────────────────────────────────────────

data0a <- all |>
  filter(Model == "GPT4") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata0a <- make_pairs(data0a, id_cols = c("Question", "Criteria")) |>
  mutate(Q_num = as.numeric(str_extract(Question, "\\d+")))

save_pairs(pairdata0a, "0a_qxc_gpt4", plot_qxc)

# ── (0b) Q x C — GPT5 ─────────────────────────────────────────────────────────

data0b <- all |>
  filter(Model == "GPT5") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata0b <- make_pairs(data0b, id_cols = c("Question", "Criteria")) |>
  mutate(Q_num = as.numeric(str_extract(Question, "\\d+")))

save_pairs(pairdata0b, "0b_qxc_gpt5", plot_qxc)

# ── (1a) Criteria — GPT4 ──────────────────────────────────────────────────────

data1a <- all_by_unique_model_criteria |>
  filter(Model == "GPT4") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata1a <- make_pairs(data1a, id_cols = "Criteria")

save_pairs(pairdata1a, "1a_c_gpt4", plot_criteria)

# ── (1b) Criteria — GPT5 ──────────────────────────────────────────────────────

data1b <- all_by_unique_model_criteria |>
  filter(Model == "GPT5") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata1b <- make_pairs(data1b, id_cols = "Criteria")

save_pairs(pairdata1b, "1b_c_gpt5", plot_criteria)

# ── (2a) Questions — GPT4 ─────────────────────────────────────────────────────

data2a <- all_by_unique_model_questions |>
  filter(Model == "GPT4") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata2a <- make_pairs(data2a, id_cols = "Question") |>
  mutate(Q_num = as.numeric(str_extract(Question, "\\d+")))

save_pairs(pairdata2a, "2a_q_gpt4", plot_questions)

# ── (2b) Questions — GPT5 ─────────────────────────────────────────────────────

data2b <- all_by_unique_model_questions |>
  filter(Model == "GPT5") |>
  pivot_longer(cols = r1:r6, names_to = "Rater", values_to = "Rating") |>
  filter(!is.na(Rating))

pairdata2b <- make_pairs(data2b, id_cols = "Question") |>
  mutate(Q_num = as.numeric(str_extract(Question, "\\d+")))

save_pairs(pairdata2b, "2b_q_gpt5", plot_questions)

# ── Keys (legends as separate PDFs) ──────────────────────────────────────────

dir.create(file.path(output_dir, "keys"), showWarnings = FALSE, recursive = TRUE)

key_theme <- theme_pubr(base_size = 11) +
  theme(
    legend.position = "right",
    legend.box      = "vertical",
    legend.title    = element_text(size = 14),
    legend.text     = element_text(size = 13)
  )

# colour bar for questions (same scale as in the plots, vertical)
q_scale <- function(bar_h = 4) {
  scale_color_viridis_c(
    option = "plasma", name = "Question", breaks = seq(1, 24, by = 4),
    guide = guide_colorbar(
      order = 1,
      theme = theme(
        legend.key.height = unit(bar_h, "cm"),
        legend.key.width  = unit(0.5, "cm")
      )
    )
  )
}

save_key <- function(p, file) {
  leg <- ggpubr::get_legend(p)
  w <- grid::convertWidth(sum(leg$widths), "cm", valueOnly = TRUE)
  h <- grid::convertHeight(sum(leg$heights), "cm", valueOnly = TRUE)
  ggsave(file, ggpubr::as_ggplot(leg),
    device = "pdf",
    width = w, height = h, units = "cm"
  )
}

# (0) Q x C: colour bar and shapes SIDE BY SIDE
key_qxc <- ggplot(key_df_qxc, aes(x, y, color = Q_num, shape = Criteria)) +
  geom_point(size = 2) +
  q_scale(bar_h = 8) +
  scale_shape_manual(
    values = criteria_shapes, name = "Criteria",
    guide = guide_legend(order = 2, override.aes = list(colour = "black", size = 3))
  ) +
  key_theme +
  theme(legend.box = "horizontal", legend.box.just = "top")

save_key(key_qxc, file.path(output_dir, "keys/key_qxc.pdf"))

# (1) Criteria: coloured points
key_df_crit <- tibble(x = 1, y = 1, Criteria = names(criteria_colors))

key_crit <- ggplot(key_df_crit, aes(x, y, color = Criteria, shape = Criteria)) +
  geom_point(size = 3) +
  scale_color_manual(values = criteria_colors, name = "Criteria") +
  scale_shape_manual(values = criteria_shapes, name = "Criteria") +
  key_theme

save_key(key_crit, file.path(output_dir, "keys/key_crit.pdf"))

# (2) Questions: colour bar only (tall, for the 2-row cell)
key_ques <- ggplot(key_df_ques, aes(x, y, color = Q_num)) +
  geom_point(size = 2) +
  q_scale(bar_h = 12) +
  key_theme

save_key(key_ques, file.path(output_dir, "keys/key_ques.pdf"))
