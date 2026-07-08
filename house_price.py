import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load Dataset

file_path = r"C:\Users\swati singh\OneDrive\Desktop\EXCEL\house_price.csv"
df = pd.read_csv(file_path)

##
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

##Feature and Target

X = df[["House_Size_sqft", "Bedrooms",
        "House_Age_years", "Distance_from_City_km"]]

y = df["House_Price_Lakhs"]

##Traun Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Train Model


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("Model R² Score:", round(r2,3))

##Market

average_price = df["House_Price_Lakhs"].mean()

low_threshold = df["House_Price_Lakhs"].quantile(0.25)
medium_threshold = df["House_Price_Lakhs"].quantile(0.50)
high_threshold = df["House_Price_Lakhs"].quantile(0.75)

# =========================================
# 6️⃣ Performance Graph
# =========================================

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# =========================================
# 7️⃣ User Prediction Function
# =========================================

def predict_house_price():
    
    print("\n====== SMART HOUSE PRICE ANALYZER ======")
    
    size = float(input("Enter House Size (sqft): "))
    bedrooms = int(input("Enter Number of Bedrooms: "))
    age = float(input("Enter House Age (years): "))
    distance = float(input("Enter Distance from City (km): "))
    
    new_data = np.array([[size, bedrooms, age, distance]])
    
    predicted_price = model.predict(new_data)[0]
    
    print("\n🏠 Predicted House Price:",
          round(predicted_price, 2), "Lakhs")
    
    print("Model Confidence (R² Score):", round(r2, 3))
    
    # =========================================
    # 📊 Price Category Classification
    # =========================================
    
    if predicted_price <= low_threshold:
        category = "LOW Category"
    elif predicted_price <= medium_threshold:
        category = "MEDIUM Category"
    elif predicted_price <= high_threshold:
        category = "HIGH Category"
    else:
        category = "LUXURY / VERY HIGH Category"
    
    print("📊 Price Category:", category)
    
    # =========================================
    # 📈 Above / Below Market Comparison
    # =========================================
    
    difference = predicted_price - average_price
    
    if difference > 0:
        print("📈 This house is ABOVE Market Average by",
              round(difference,2), "Lakhs")
    else:
        print("📉 This house is BELOW Market Average by",
              round(abs(difference),2), "Lakhs")
    
    print("Market Average Price:",
          round(average_price,2), "Lakhs")

# Call Function
predict_house_price()
