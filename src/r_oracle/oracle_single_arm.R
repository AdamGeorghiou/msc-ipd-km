#!/usr/bin/env Rscript

# R Oracle Script for Single Arm IPD Reconstruction
# This produces the "ground truth" we'll compare Python against

library(IPDfromKM)

# Load sample data
radio_curve <- read.csv("data/examples/radio_curve.csv")
at_risk_radio <- read.csv("data/examples/at_risk_radio.csv")

cat("Loaded data:\n")
cat(sprintf("  Curve points: %d rows\n", nrow(radio_curve)))
cat(sprintf("  At-risk table: %d time points\n", nrow(at_risk_radio)))

# Preprocess
preprocessed <- preprocess(
  dat = radio_curve,
  trisk = at_risk_radio$time,
  nrisk = at_risk_radio$n_risk,
  maxy = 100 # Data is in percentages (0-100)
)

cat("\n✅ Preprocessing complete\n")

ipd_result <- getIPD(
  prep = preprocessed,
  armID = 0,
  tot.events = NULL
)

cat("✅ IPD reconstruction complete\n")

# Save result
output_ipd <- ipd_result$IPD

# Standardize column name to 'arm' (R may use 'treat')
if ("treat" %in% colnames(output_ipd)) {
  colnames(output_ipd)[colnames(output_ipd) == "treat"] <- "arm"
}

write.csv(output_ipd, "results/oracle_radio.csv", row.names = FALSE)

# Print summary
cat("\n", paste(rep("=", 50), collapse = ""), "\n")
cat("R Oracle Summary\n")
cat(paste(rep("=", 50), collapse = ""), "\n")
cat(sprintf("Output saved to: results/oracle_radio.csv\n"))
cat(sprintf("Reconstructed: %d patient records\n", nrow(output_ipd)))
cat(sprintf("Events: %d\n", sum(output_ipd$status == 1)))
cat(sprintf("Censored: %d\n", sum(output_ipd$status == 0)))
cat(paste(rep("=", 50), collapse = ""), "\n")
