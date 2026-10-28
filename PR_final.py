import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("data", exist_ok=True)
os.makedirs("graphs", exist_ok=True)


class CovidDataAnalyzer:

    def __init__(self):
        self.data = None

    def load_dataset(self):
        file_path = "data/covid_data.csv"

        try:
            self.data = pd.read_csv(file_path)

            print("\nDataset loaded successfully!")
            print("Total rows:", self.data.shape[0])
            print("Total columns:", self.data.shape[1])

        except FileNotFoundError:
            print("\nDataset file not found!")
            print("Place covid_data.csv inside the data folder.")

        except Exception as error:
            print("Error loading dataset:", error)

    def check_data(self):
        if self.data is None:
            print("\nPlease load the dataset first.")
            return False

        return True

    def explore_data(self):
        if not self.check_data():
            return

        print("\nFIRST FIVE ROWS")
        print(self.data.head())

        print("\nCOLUMN NAMES")
        print(self.data.columns.tolist())

        print("\nDATA INFORMATION")
        self.data.info()

        print("\nDESCRIPTIVE STATISTICS")
        print(self.data.describe())

    def handle_missing_data(self):
        if not self.check_data():
            return

        print("\nMissing values before cleaning:")
        print(self.data.isnull().sum().head(20))

        duplicate_count = self.data.duplicated().sum()
        print("\nDuplicate rows:", duplicate_count)

        self.data = self.data.drop_duplicates()

        for column in ["total_cases", "total_deaths"]:
            if column in self.data.columns:
                self.data[column] = self.data[column].fillna(0)

        print("\nMissing-value handling completed.")
        print("Rows remaining:", len(self.data))

    def country_analysis(self):
        if not self.check_data():
            return

        required = ["location", "date", "total_cases", "total_deaths"]

        if not all(column in self.data.columns for column in required):
            print("Required columns are missing from the dataset.")
            return

        country = input("Enter country name (example: India): ").strip()

        country_data = self.data[
            self.data["location"].astype(str).str.lower() == country.lower()
        ].copy()

        if country_data.empty:
            print("Country not found. Check the spelling.")
            return

        country_data["date"] = pd.to_datetime(
            country_data["date"], errors="coerce"
        )

        country_data = country_data.dropna(subset=["date"])
        country_data = country_data.sort_values("date")

        if country_data.empty:
            print("No valid date records found.")
            return

        latest = country_data.iloc[-1]

        print("\nCOUNTRY SUMMARY")
        print("Country:", country)
        print("Latest available date:", latest["date"].date())
        print("Total reported cases:", latest["total_cases"])
        print("Total reported deaths:", latest["total_deaths"])

    def compare_countries(self):
        if not self.check_data():
            return

        required = ["location", "date", "total_cases"]

        if not all(column in self.data.columns for column in required):
            print("Required columns are missing from the dataset.")
            return

        countries = input(
            "Enter country names separated by commas "
            "(example: India,United States,Brazil): "
        )

        country_list = [
            country.strip()
            for country in countries.split(",")
            if country.strip()
        ]

        if not country_list:
            print("Please enter at least one country.")
            return

        selected = self.data[
            self.data["location"].astype(str).str.lower().isin(
                [country.lower() for country in country_list]
            )
        ].copy()

        if selected.empty:
            print("No matching countries found.")
            return

        selected["date"] = pd.to_datetime(
            selected["date"], errors="coerce"
        )

        selected = selected.dropna(subset=["date"])

        if selected.empty:
            print("No valid date records found.")
            return

        summary = (
            selected.sort_values("date")
            .groupby("location", as_index=False)
            .tail(1)
        )

        summary = summary[["location", "date", "total_cases"]]

        print("\nCOUNTRY COMPARISON")
        print(summary.to_string(index=False))

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=summary,
            x="location",
            y="total_cases"
        )

        plt.title("Latest Reported COVID-19 Cases by Country")
        plt.xlabel("Country")
        plt.ylabel("Reported Total Cases")
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig("graphs/country_comparison.png")
        plt.show()
        plt.close()

        print("Graph saved: graphs/country_comparison.png")

    def plot_country_trends(self):
        if not self.check_data():
            return

        required = ["location", "date", "total_cases", "total_deaths"]

        if not all(column in self.data.columns for column in required):
            print("Required columns are missing from the dataset.")
            return

        country = input("Enter country name (example: India): ").strip()

        country_data = self.data[
            self.data["location"].astype(str).str.lower() == country.lower()
        ].copy()

        if country_data.empty:
            print("Country not found.")
            return

        country_data["date"] = pd.to_datetime(
            country_data["date"], errors="coerce"
        )

        country_data = country_data.dropna(subset=["date"])
        country_data = country_data.sort_values("date")

        if country_data.empty:
            print("No valid date records found.")
            return

        for column in ["total_cases", "total_deaths"]:
            country_data[column] = pd.to_numeric(
                country_data[column], errors="coerce"
            )

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=country_data,
            x="date",
            y="total_cases",
            label="Total reported cases"
        )

        plt.title(f"COVID-19 Cases Over Time - {country}")
        plt.xlabel("Date")
        plt.ylabel("Total reported cases")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("graphs/covid_cases_trend.png")
        plt.show()
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=country_data,
            x="date",
            y="total_deaths",
            label="Total reported deaths"
        )

        plt.title(f"COVID-19 Deaths Over Time - {country}")
        plt.xlabel("Date")
        plt.ylabel("Total reported deaths")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig("graphs/covid_deaths_trend.png")
        plt.show()
        plt.close()

        print("Case graph saved: graphs/covid_cases_trend.png")
        print("Death graph saved: graphs/covid_deaths_trend.png")

    def display_summary(self):
        if not self.check_data():
            return

        print("\nDATASET SUMMARY")
        print("Total records:", len(self.data))
        print("Total columns:", len(self.data.columns))

        if "location" in self.data.columns:
            print("Number of locations:", self.data["location"].nunique())

        for column in ["total_cases", "total_deaths"]:
            if column in self.data.columns:
                values = pd.to_numeric(
                    self.data[column], errors="coerce"
                )

                print(f"\n{column} summary:")
                print(values.describe())

    def run(self):
        while True:
            print("\n" + "=" * 45)
            print(" COVID-19 DATA ANALYSIS AND VISUALIZATION")
            print("=" * 45)
            print("1. Load Dataset")
            print("2. Explore Dataset")
            print("3. Handle Missing Data")
            print("4. Analyze a Country")
            print("5. Compare Countries")
            print("6. Generate Cases and Deaths Graphs")
            print("7. Display Summary")
            print("8. Exit")

            choice = input("\nEnter your choice: ").strip()

            if choice == "1":
                self.load_dataset()
            elif choice == "2":
                self.explore_data()
            elif choice == "3":
                self.handle_missing_data()
            elif choice == "4":
                self.country_analysis()
            elif choice == "5":
                self.compare_countries()
            elif choice == "6":
                self.plot_country_trends()
            elif choice == "7":
                self.display_summary()
            elif choice == "8":
                print("Thank you for using the COVID-19 Analyzer!")
                break
            else:
                print("Invalid choice. Please enter a number from 1 to 8.")


if __name__ == "__main__":
    analyzer = CovidDataAnalyzer()
    analyzer.run()