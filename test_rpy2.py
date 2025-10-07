import rpy2.robjects as ro
from rpy2.robjects.packages import importr

# Test basic R
print("Testing R connection...")
ro.r('print("Hello from R!")')

# Test IPDfromKM package
print("\nTesting IPDfromKM package...")
try:
    ipd = importr('IPDfromKM')
    print("✅ IPDfromKM loaded successfully!")
    print(f"Package functions: {dir(ipd)[:5]}...")  # Show first 5 functions
except Exception as e:
    print(f"❌ Error loading IPDfromKM: {e}")