import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('C:\\Users\\abhis\\Downloads\\train.csv')

print("Original Dataset:")
print(df.head())

if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
else:
    print("Column 'Age' not found in the dataset.")

if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\nColumn Names after handling missing values:", df.columns)

available_columns = df.columns.tolist()
print("\nAvailable Columns:", available_columns)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize']

features = [feature for feature in features if feature in available_columns]

print("\nSelected Features:", features)

try:
    X = df[features]
    y = df['Survived'] 

    X = pd.get_dummies(X, columns=['Sex', 'Embarked'], drop_first=True)

    X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{classification_rep}")

except Exception as e:
    print("\nError:", e)