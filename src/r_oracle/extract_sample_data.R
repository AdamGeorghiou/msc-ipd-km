# Load IPDfromKM package
library(IPDfromKM)

# Access built-in Radiationdata
data(Radiationdata)

# Inspect the data structure first
print("Data structure:")
print(names(Radiationdata))
print(paste("trisk length:", length(Radiationdata$trisk)))
print(paste("nrisk.radio length:", length(Radiationdata$nrisk.radio)))
print(paste("nrisk.radioplus length:", length(Radiationdata$nrisk.radioplus)))

# Save curve data
write.csv(Radiationdata$radio, 
          "data/examples/radio_curve.csv", 
          row.names = FALSE)

write.csv(Radiationdata$radioplus, 
          "data/examples/radioplus_curve.csv", 
          row.names = FALSE)

# Save at-risk information - handle different lengths
# Radio arm (use only the first 6 time points)
at_risk_radio <- data.frame(
  time = Radiationdata$trisk[1:length(Radiationdata$nrisk.radio)],
  n_risk = Radiationdata$nrisk.radio
)

at_risk_radioplus <- data.frame(
  time = Radiationdata$trisk[1:length(Radiationdata$nrisk.radioplus)],
  n_risk = Radiationdata$nrisk.radioplus
)

write.csv(at_risk_radio, 
          "data/examples/at_risk_radio.csv", 
          row.names = FALSE)

write.csv(at_risk_radioplus, 
          "data/examples/at_risk_radioplus.csv", 
          row.names = FALSE)

# Also save combined (for reference)
max_len <- max(length(Radiationdata$nrisk.radio), 
               length(Radiationdata$nrisk.radioplus))

at_risk_combined <- data.frame(
  time = Radiationdata$trisk[1:max_len]
)

# Add radio data (pad with NA if needed)
at_risk_combined$n_risk_radio <- c(Radiationdata$nrisk.radio, 
                                    rep(NA, max_len - length(Radiationdata$nrisk.radio)))

# Add radioplus data (pad with NA if needed)
at_risk_combined$n_risk_radioplus <- c(Radiationdata$nrisk.radioplus, 
                                        rep(NA, max_len - length(Radiationdata$nrisk.radioplus)))

write.csv(at_risk_combined, 
          "data/examples/at_risk_combined.csv", 
          row.names = FALSE)

cat("\n✅ Sample data extracted successfully!\n")
cat("Files saved:\n")
cat("  - data/examples/radio_curve.csv\n")
cat("  - data/examples/radioplus_curve.csv\n")
cat("  - data/examples/at_risk_radio.csv\n")
cat("  - data/examples/at_r