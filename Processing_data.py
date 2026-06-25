{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "353c7dbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np  \n",
    "df_budget = pd.read_csv(\"Budget_Noisy.csv\")\n",
    "df_expenses = pd.read_csv(\"Expenses_Noisy.csv\")\n",
    "df_income = pd.read_csv(\"Income_Noisy.csv\")\n",
    "# Xóa dòng trùng nhau \n",
    "df_expenses.drop_duplicates(inplace = True)\n",
    "# Định dạng ngày tháng\n",
    "df_expenses['date_time'] = pd.to_datetime(df_expenses['date_time'], errors='coerce').ffill().bfill()\n",
    "df_income['date_time'] = pd.to_datetime(df_income['date_time'], errors='coerce').ffill().bfill()\n",
    "# Xử lí cac giá trị bị khuyết trống thiếu ở cột Amount với giá trị ở giữa (Median) - ở cột Category với chữ xuất hiện nhiều nhất (Mode) \n",
    "df_expenses['amount'] = df_expenses['amount'].fillna(df_expenses['amount'].median())\n",
    "df_expenses['category'] = df_expenses['category'].fillna(df_expenses['category'].mode()[0])\n",
    "df_income['amount'] = df_income['amount'].fillna(df_income['amount'].median())\n",
    "df_income['category'] = df_income['category'].fillna(df_income['category'].mode()[0])\n",
    "df_budget['amount'] = df_budget['amount'].fillna(df_budget['amount'].median())\n",
    "# XUẤT FILE CLEANED\n",
    "df_budget.to_csv(\"Budget_CLEANED.csv\", index = False)\n",
    "df_income.to_csv(\"Income_CLEANED.csv\", index = False)\n",
    "df_expenses.to_csv(\"Expenses_CLEANED.csv\", index = False )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
