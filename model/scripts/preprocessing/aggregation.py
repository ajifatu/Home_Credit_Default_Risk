import pandas as pd

def aggregate_bureau_balance_data(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    grouped_status = bureau_balance.groupby(['SK_ID_BUREAU', 'STATUS']).size()
    pivoted_status = grouped_status.unstack(fill_value=0)
    pivoted_status.columns = [f'STATUS_{col}' for col in pivoted_status.columns]
    aggregated_months_balance = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].agg(['count']).reset_index()
    aggregated_months_balance.rename(columns={'count': 'TOTAL_MONTHS'}, inplace=True)
    pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(
        axis=1)
    pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum( axis=1)
    pivoted_status = pivoted_status.reset_index().merge(aggregated_months_balance, on='SK_ID_BUREAU', how='left')
    pivoted_status = pivoted_status.reset_index().merge(aggregated_months_balance, on='SK_ID_BUREAU', how='left',
                                                        suffixes=('', '_y'))
    pivoted_status.drop(columns=['TOTAL_MONTHS_y'], inplace=True)
    pivoted_status['STATUS_DELAYED'] = pivoted_status[['STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5']].sum(
        axis=1)
    pivoted_status['CREDIT_STATUS'] = pivoted_status.apply(
        lambda x: 'Completed' if x['STATUS_C'] > x['STATUS_DELAYED'] and x['STATUS_C'] > x['STATUS_X'] else
        'Delayed' if x['STATUS_DELAYED'] > x['STATUS_C'] and x['STATUS_DELAYED'] > x['STATUS_X'] else
        'X' if x['STATUS_X'] > x['STATUS_C'] and x['STATUS_X'] > x['STATUS_DELAYED'] else
        'No Data', axis=1
    )
    aggregated_bureau = pivoted_status
    aggregated_bureau.head()
    return aggregated_bureau

def aggregate_bureau_balance_and_bureau_data(bureau: pd.DataFrame, bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bureau_balance = aggregate_bureau_balance_data(bureau_balance)
    left_tables = bureau.merge(bureau_balance, on='SK_ID_BUREAU', how='left')
    categorical_columns = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE', 'CREDIT_STATUS']
    numerical_columns = left_tables.select_dtypes(include=['int64', 'float64']).columns.tolist()
    individual_aggregations = {}
    for col in categorical_columns:
        grouped_categorical = left_tables.groupby('SK_ID_CURR')[col].value_counts().unstack(fill_value=0)
        grouped_categorical.columns = [f'{col}_{col_val}' for col_val in grouped_categorical.columns]
        grouped_categorical.reset_index(inplace=True)
        individual_aggregations[col] = grouped_categorical

    aggregated_Days_Credit = left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT'].agg(['max']).reset_index()
    aggregated_Days_Credit.rename(columns={'max': 'DAYS_CREDIT_MAX'}, inplace=True)
    aggregated_CREDIT_DAY_OVERDUE = left_tables.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].agg(['max']).reset_index()
    aggregated_CREDIT_DAY_OVERDUE.rename(columns={'max': 'CREDIT_DAY_OVERDUE_MAX'}, inplace=True)
    aggregated_DAYS_CREDIT_ENDDATE = left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].agg(['min']).reset_index()
    aggregated_DAYS_CREDIT_ENDDATE.rename(columns={'min': 'DAYS_CREDIT_ENDDATE_MIN'}, inplace=True)
    aggregated_DAYS_ENDDATE_FACT = left_tables.groupby('SK_ID_CURR')['DAYS_ENDDATE_FACT'].agg(['max']).reset_index()
    aggregated_DAYS_ENDDATE_FACT.rename(columns={'max': 'DAYS_ENDDATE_FACT_MAX'}, inplace=True)
    aggregated_AMT_CREDIT_MAX_OVERDUE = left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].agg(['max']).reset_index()
    aggregated_AMT_CREDIT_MAX_OVERDUE.rename(columns={'max': 'AMT_CREDIT_MAX_OVERDUE_MAX'}, inplace=True)
    aggregated_CNT_CREDIT_PROLONG = left_tables.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].agg(['max']).reset_index()
    aggregated_CNT_CREDIT_PROLONG.rename(columns={'max': 'CNT_CREDIT_PROLONG_MAX'}, inplace=True)
    aggregated_AMT_CREDIT_SUM = left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].agg(['sum']).reset_index()
    aggregated_AMT_CREDIT_SUM.rename(columns={'sum': 'AMT_CREDIT_SUM_SUM', 'mean': 'AMT_CREDIT_SUM_MEAN'}, inplace=True)
    aggregated_AMT_CREDIT_SUM_DEBT = left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].agg(['sum']).reset_index()
    aggregated_AMT_CREDIT_SUM_DEBT.rename(columns={'sum': 'AMT_CREDIT_SUM_DEBT_SUM'}, inplace=True)
    aggregated_AMT_CREDIT_SUM_LIMIT = left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].agg(['mean']).reset_index()
    aggregated_AMT_CREDIT_SUM_LIMIT.rename(columns={'mean': 'AMT_CREDIT_SUM_LIMIT_MEAN'}, inplace=True)
    aggregated_AMT_CREDIT_SUM_LIMIT['AMT_CREDIT_SUM_LIMIT_MEAN'].fillna(0, inplace=True)
    aggregated_AMT_CREDIT_SUM_OVERDUE = left_tables.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].agg(['max']).reset_index()
    aggregated_AMT_CREDIT_SUM_OVERDUE.rename(columns={'max': 'AMT_CREDIT_SUM_OVERDUE_MAX'}, inplace=True)
    aggregated_DAYS_CREDIT_UPDATE = left_tables.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].agg(['min']).reset_index()
    aggregated_DAYS_CREDIT_UPDATE.rename(columns={'min': 'DAYS_CREDIT_UPDATE_MIN'}, inplace=True)
    aggregated_AMT_ANNUITY = left_tables.groupby('SK_ID_CURR')['AMT_ANNUITY'].agg(['mean']).reset_index()
    aggregated_AMT_ANNUITY.rename(columns={'mean': 'AMT_ANNUITY_MEAN'}, inplace=True)
    aggregated_AMT_ANNUITY['AMT_ANNUITY_MEAN'].fillna(0, inplace=True)
    aggregated_status_0 = left_tables.groupby('SK_ID_CURR')['STATUS_0'].agg('sum').reset_index()
    aggregated_status_1 = left_tables.groupby('SK_ID_CURR')['STATUS_1'].agg('sum').reset_index()
    aggregated_status_2 = left_tables.groupby('SK_ID_CURR')['STATUS_2'].agg('count').reset_index()
    aggregated_status_3 = left_tables.groupby('SK_ID_CURR')['STATUS_3'].agg('count').reset_index()
    aggregated_status_4 = left_tables.groupby('SK_ID_CURR')['STATUS_4'].agg('count').reset_index()
    aggregated_status_5 = left_tables.groupby('SK_ID_CURR')['STATUS_5'].agg('count').reset_index()
    aggregated_status_X = left_tables.groupby('SK_ID_CURR')['STATUS_X'].agg('sum').reset_index()
    aggregated_status_C = left_tables.groupby('SK_ID_CURR')['STATUS_C'].agg('sum').reset_index()
    aggregated_TOTAL_MONTHS = left_tables.groupby('SK_ID_CURR')['TOTAL_MONTHS'].agg('count').reset_index()
    aggregated_STATUS_DELAYED = left_tables.groupby('SK_ID_CURR')['STATUS_DELAYED'].agg('sum').reset_index()
    aggregated_dataframes = [
        aggregated_Days_Credit,
        aggregated_CREDIT_DAY_OVERDUE,
        aggregated_DAYS_CREDIT_ENDDATE,
        aggregated_DAYS_ENDDATE_FACT,
        aggregated_AMT_CREDIT_MAX_OVERDUE,
        aggregated_CNT_CREDIT_PROLONG,
        aggregated_AMT_CREDIT_SUM,
        aggregated_AMT_CREDIT_SUM_DEBT,
        aggregated_AMT_CREDIT_SUM_LIMIT,
        aggregated_AMT_CREDIT_SUM_OVERDUE,
        aggregated_DAYS_CREDIT_UPDATE,
        aggregated_AMT_ANNUITY,
        aggregated_status_0,
        aggregated_status_1,
        aggregated_status_2,
        aggregated_status_3,
        aggregated_status_4,
        aggregated_status_5,
        aggregated_status_C,
        aggregated_status_X, aggregated_STATUS_DELAYED,
        aggregated_TOTAL_MONTHS
    ]
    aggregated_final_data = pd.concat(aggregated_dataframes, axis=1)
    aggregated_final_data = aggregated_final_data.loc[:, ~aggregated_final_data.columns.duplicated()]
    aggregated_left_tables = aggregated_final_data
    aggregated_left_tables.head()
    return aggregated_left_tables

def aggregated_previous_application(previous_application:pd.DataFrame)->pd.DataFrame:
    columns_to_drop = [
        "SK_ID_PREV",
        "WEEKDAY_APPR_PROCESS_START",
        "HOUR_APPR_PROCESS_START",
        "FLAG_LAST_APPL_PER_CONTRACT",
        "NFLAG_LAST_APPL_IN_DAY",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_PAYMENT_TYPE",
        "NAME_TYPE_SUITE",
        "NAME_CLIENT_TYPE",
        "NAME_PRODUCT_TYPE",
        "CHANNEL_TYPE",
        "SELLERPLACE_AREA",
        "NAME_SELLER_INDUSTRY",
        "PRODUCT_COMBINATION",
        "NFLAG_INSURED_ON_APPROVAL"
    ]
    previous_application = previous_application.drop(columns=[col for col in columns_to_drop if col in previous_application.columns])
    aggregations = {
        'AMT_APPLICATION': 'mean',
        'AMT_ANNUITY': 'mean',
        'AMT_CREDIT': 'mean',
        'AMT_DOWN_PAYMENT': 'max',
        'AMT_GOODS_PRICE': 'mean',
        'RATE_DOWN_PAYMENT': 'mean',
        'RATE_INTEREST_PRIMARY': 'mean',
        'DAYS_DECISION': 'min',
        'CNT_PAYMENT': 'max',
        'DAYS_FIRST_DRAWING': 'min',
        'DAYS_FIRST_DUE': 'min',
        'DAYS_LAST_DUE': 'max',
        'DAYS_TERMINATION': 'max',
        'NAME_CONTRACT_TYPE': 'count',
        'NAME_CONTRACT_STATUS': 'count',
        'NAME_GOODS_CATEGORY': 'count',
        'NAME_PORTFOLIO': 'count',
        'NAME_YIELD_GROUP': 'count',
    }
    aggregated_previous_application = previous_application.groupby('SK_ID_CURR').agg(aggregations)
    aggregated_previous_application.columns = [
        f"{col}_{aggregations[col].upper()}" for col in aggregated_previous_application.columns
    ]
    aggregated_previous_application.reset_index(inplace=True)
    return aggregated_previous_application

def aggregated_pos_cach_balance(pos_cash_balance:pd.DataFrame)->pd.DataFrame:
    aggregations = {
        'MONTHS_BALANCE': 'count',
        'CNT_INSTALMENT': ['mean', 'max'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum'],
        'SK_DPD': ['mean', 'max'],
        'SK_DPD_DEF': 'mean'
    }
    pos_agg = pos_cash_balance.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in pos_agg.columns]
    contract_statuses = ['Active', 'Completed', 'Signed', 'Demand']
    for status in contract_statuses:
        pos_agg[f'NAME_CONTRACT_STATUS_{status}'] = (
            (pos_cash_balance['NAME_CONTRACT_STATUS'] == status)
            .groupby(pos_cash_balance['SK_ID_CURR']).transform('sum')
        )
    pos_agg = pos_agg.reset_index()
    aggregated_POS_CACH_balance = pos_agg
    return aggregated_POS_CACH_balance

def aggregated_installments_payments(installments_payments:pd.DataFrame)->pd.DataFrame:
    installments_agg = installments_payments.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['count'],
        'NUM_INSTALMENT_NUMBER': ['max'],
        'DAYS_INSTALMENT': ['min'],
        'DAYS_ENTRY_PAYMENT': ['min'],
        'AMT_INSTALMENT': ['sum'],
        'AMT_PAYMENT': ['sum']
    }).reset_index()

    installments_agg.columns = ['_'.join(col).strip('_') for col in installments_agg.columns]
    aggregated_installments_payments = installments_agg
    return aggregated_installments_payments

def aggregated_credit_card_balance(credit_card_balance:pd.DataFrame)->pd.DataFrame:
    credit_card_agg = credit_card_balance.groupby('SK_ID_CURR').agg({
        'MONTHS_BALANCE': ['count'],
        'AMT_BALANCE': ['mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['sum'],
        'AMT_DRAWINGS_CURRENT': ['sum'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['sum'],
        'AMT_DRAWINGS_POS_CURRENT': ['sum'],
        'AMT_INST_MIN_REGULARITY': ['mean'],
        'AMT_PAYMENT_CURRENT': ['sum'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['sum'],
        'AMT_RECEIVABLE_PRINCIPAL': ['max'],
        'AMT_RECIVABLE': ['max'],
        'AMT_TOTAL_RECEIVABLE': ['max'],
        'CNT_DRAWINGS_ATM_CURRENT': ['sum'],
        'CNT_DRAWINGS_CURRENT': ['sum'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['sum'],
        'CNT_DRAWINGS_POS_CURRENT': ['sum'],
        'CNT_INSTALMENT_MATURE_CUM': ['max'],
        'SK_DPD': ['max'],
        'SK_DPD_DEF': ['max'],
        'NAME_CONTRACT_STATUS': ['count']
    }).reset_index()

    credit_card_agg.columns = ['_'.join(col).strip('_') for col in credit_card_agg.columns]
    aggregated_credit_card_balance = credit_card_agg
    return aggregated_credit_card_balance

if __name__ == '__main__':
    import pandas as pd
    import os

    print(os.getcwd())

    bureau_balance_df = pd.read_csv('bureau_balance.csv')
    bureau_df = pd.read_csv('bureau.csv')
    previous_application_df = pd.read_csv('previous_application.csv')
    pos_cash_balance_df = pd.read_csv('pos_cash_balance.csv')
    installments_payments_df = pd.read_csv('installments_payments.csv')
    credit_card_balance_df = pd.read_csv('credit_card_balance.csv')

    aggregated_bureau_balance = aggregate_bureau_balance_data(bureau_balance_df)
    aggregated_bureau = aggregate_bureau_balance_and_bureau_data(bureau_df, bureau_balance_df)
    aggregated_prev_app = aggregated_previous_application(previous_application_df)
    aggregated_pos_cash = aggregated_pos_cach_balance(pos_cash_balance_df)
    aggregated_installments = aggregated_installments_payments(installments_payments_df)
    aggregated_credit_card = aggregated_credit_card_balance(credit_card_balance_df)

    print("aggregating data finished")
    print("let's print the first 5 rows of each dataframe")
    for df in [aggregated_bureau_balance, aggregated_bureau, aggregated_prev_app, aggregated_pos_cash, aggregated_installments, aggregated_credit_card]:
        print(df.head())
        print("\n")
