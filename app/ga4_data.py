import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import os
from .generate_dummy_data import generate_dummy_data

def generate_scroll_lp_dummy_data(num_events=5000, num_days=30):
    """
    generate_dummy_dataのラッパー関数
    """
    return generate_dummy_data(num_events, num_days)

@st.cache_data
def fetch_ga4_data():
    """
    BigQueryからGA4データを取得する関数
    Secretsに情報がない場合やエラーが発生した場合はダミーデータを返す
    """
    try:
        # StreamlitのSecretsから認証情報を取得
        if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
            gcp_service_account_str = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
            credentials_info = json.loads(gcp_service_account_str)
            credentials = service_account.Credentials.from_service_account_info(credentials_info)
            project_id = credentials.project_id

            # BigQueryクライアントを初期化
            client = bigquery.Client(credentials=credentials, project=project_id)

            # テーブル名 (プロジェクトIDを動的に使用)
            table_name = f"{project_id}.shungene_dataset.swipelp_events"

            # SQLクエリを作成
            query = f"SELECT * FROM `{table_name}`"

            # クエリを実行してDataFrameに読み込む
            df = client.query(query).to_dataframe()

            # タイムスタンプと日付の列をdatetime型に変換
            if 'event_timestamp' in df.columns:
                df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
            if 'event_date' in df.columns:
                df['event_date'] = pd.to_datetime(df['event_date'])
            
            st.toast("BigQueryからデータを正常に読み込みました。", icon="✅")
            return df
        else:
            # Secretsがない場合
            st.warning("GCP_SERVICE_ACCOUNT_JSONがSecretsに設定されていません。ダミーデータを使用します。")
            return generate_scroll_lp_dummy_data()

    except Exception as e:
        st.warning(f"BigQueryからのデータ取得に失敗しました: {e}。ダミーデータを使用します。")
        return generate_scroll_lp_dummy_data()
