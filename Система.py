import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from rdkit.Chem import Descriptors
from rdkit import Chem
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBClassifier
import gdown
import requests

class KerasNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim=None, lr=0.001, epochs=30, batch_size=32, verbose=0,
                 dropout_rate=0.3, l2_lambda=0.001):
        self.input_dim = input_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.l2_lambda = l2_lambda
        self.model_ = None

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, activation="relu", input_dim=self.input_dim,
                        kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(256, activation="relu", kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(128, activation="relu", kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation="relu", kernel_regularizer=l2(self.l2_lambda)))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(optimizer=Nadam(learning_rate=self.lr),
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        return model

    def fit(self, X, y):
        self.model_ = self._build_model()
        self.model_.fit(X, y, epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.verbose)
        return self

    def predict(self, X):
        pred = self.model_.predict(X)
        return (pred.ravel() >= 0.5).astype(int)

    def predict_proba(self, X):
        pred = self.model_.predict(X)
        return np.vstack([1 - pred.ravel(), pred.ravel()]).T

MODELS_PATH = "models"
os.makedirs(MODELS_PATH, exist_ok=True)

def download_model_from_drive(file_id, filename):
    url = f"https://drive.google.com/uc?id={file_id}"
    out_path = os.path.join(MODELS_PATH, filename)
    if not os.path.exists(out_path):
        gdown.download(url, out_path, quiet=False)
    return out_path

# Словник доступних моделей
available_models = {
    "Random Forest": {
        "Mordred": {
            "Всі": "rf_pipeline_bundle_RF_ALL Mordred.pkl",
            "Аліфатичні ациклічні": "rf_pipeline_bundle_Aliphatic_acyclic_Mordred.pkl",
            "Аліфатичні гетероциклічні": "rf_pipeline_bundle_Aliphatic heteromono(poly)cyclic Mordred.pkl",
            "Ароматичні гетероциклічні": "rf_pipeline_bundle_Aromatic heteromono(poly)cyclic Mordred.pkl",
            "Ароматичні гомоциклічні": "rf_pipeline_bundle_Aromatic homomono(poly)cyclic Mordred.pkl"
        },
        "PaDEL": {
            "Всі": "rf_pipeline_bundle_RF_ALL PaDel.pkl",
            "Аліфатичні ациклічні": "rf_pipeline_bundle_Aliphatic acyclic_PaDel.pkl",
            "Аліфатичні гетероциклічні": "rf_pipeline_bundle_Aliphatic heteromono(poly)cyclic Class PaDel.pkl",
            "Ароматичні гетероциклічні": "rf_pipeline_bundle_Aromatic heteromono(poly) PaDel.pkl",
            "Ароматичні гомоциклічні": "rf_pipeline_bundle_Aromatic homomono(poly)cyclic Class PaDel.pkl"
        },
        "RDKit": {
            "Всі": "rf_pipeline_bundle_RF_ALL_RDkit.pkl",
            "Аліфатичні ациклічні": "rf_pipeline_bundle_Aliphatic acyclic_RDkit.pkl",
            "Аліфатичні гетероциклічні": "rf_pipeline_bundle_Aliphatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гетероциклічні": "rf_pipeline_bundle_Aromatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гомоциклічні": "rf_pipeline_bundle_Aromatic homomono(poly)cyclic RDkit.pkl"
        }
    },
    "Boosting": {
        "Mordred": {
            "Всі": "xgb_pipeline_bundle_ALL_Mordred.pkl",
            "Аліфатичні ациклічні": "xgb_pipeline_bundle_Aliphatic heteromono(poly)cyclic Class Mordred.pkl",
            "Аліфатичні гетероциклічні": "xgb_pipeline_bundle_Aliphatic heteromono(poly)cyclic Class Mordred.pkl",
            "Ароматичні гетероциклічні": "xgb_pipeline_bundle_Aromatic heteromono(poly) cyclic Mordred.pkl",
            "Ароматичні гомоциклічні": "xgb_pipeline_bundle_Aromatic homomono(poly)cyclic Mordred.pkl"
        },
        "PaDEL": {
            "Всі": "xgb_pipeline_bundle_ALL_PaDell.pkl",
            "Аліфатичні ациклічні": "xgb_pipeline_bundle_Aliphatic acyclic PaDelt.pkl",
            "Аліфатичні гетероциклічні": "xgb_pipeline_bundle_Aliphatic heteromono(poly)cyclic Class PaDel.pkl",
            "Ароматичні гетероциклічні": "xgb_pipeline_bundle_Aromatic heteromono(poly) cyclic PaDelt.pkl",
            "Ароматичні гомоциклічні": "xgb_pipeline_bundle_Aromatic homomono(poly)cyclic PaDelt.pkl"
        },
        "RDKit": {
            "Всі": "xgb_pipeline_bundle_ALL_RDkit.pkl",
            "Аліфатичні ациклічні": "xgb_pipeline_bundle_Aliphatic acyclic RDkit.pkl",
            "Аліфатичні гетероциклічні": "xgb_pipeline_bundle_Aliphatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гетероциклічні": "xgb_pipeline_bundle_Aromatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гомоциклічні": "xgb_pipeline_bundle_Aromatic homomono(poly)cyclic RDkit.pkl" }
    },
    "Neural Network": {
        "Mordred": {
            "Всі": "nn_ALL_Mordred.pkl",
            "Аліфатичні ациклічні": "nn_Aliphatic acyclic Mordred.pkl",
            "Аліфатичні гетероциклічні": "nn_Aliphatic heteromono(poly)cyclic Class Mordred.pkl",
            "Ароматичні гетероциклічні": "nn_Aromatic heteromono(poly)cyclic Class Mordred.pkl",
            "Ароматичні гомоциклічні": "nn_Aromatic homomono(poly)cyclic Class Mordred.pkl"
        },
        "PaDEL": {
            "Всі": "nn_ALL PaDel.pkl",
            "Аліфатичні ациклічні": "nn_Aliphatic acyclic PaDel.pkl",
            "Аліфатичні гетероциклічні": "nn_Aliphatic heteromono(poly)cyclic PaDel.pkl",
            "Ароматичні гетероциклічні": "nn_Aromatic heteromono(poly) cyclic Class PaDel.pkl",
            "Ароматичні гомоциклічні": "nn_Aromatic homomono(poly)cyclic Class PaDel.pkl"
        },
        "RDKit": {
            "Всі": "nn_ALL_RDkit.pkl",
            "Аліфатичні ациклічні": "nn_Aliphatic acyclic RDkit.pkl",
            "Аліфатичні гетероциклічні": "nn_Aliphatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гетероциклічні": "nn_Aromatic heteromono(poly)cyclic RDkit.pkl",
            "Ароматичні гомоциклічні": "nn_Aromatic homomono(poly)cyclic RDkit.pkl"
        }
    }
}

# Словник доступних моделей
available_models2 = {
    "Random Forest": {
        "Mordred": {
            "Всі": "1pv4secP_mHC8f7eWeBt29k1VLnn7XJDe",
            "Аліфатичні ациклічні": "1wZXw5YwsFtd7nejxjde1f-aUgI2ymP_S",
            "Аліфатичні гетероциклічні": "118X6QcqV6i-Q5fEoS7sMK4Ca2kcun9dd",
            "Ароматичні гетероциклічні": "1aM4RUrVvJiSqDzdx7ROpFn-1dttjtNab",
            "Ароматичні гомоциклічні": "1JKuOrpQU5t-0n2bLRmYZF7lHd-YNagzz"
        },
        "PaDEL": {
            "Всі": "1VVcD1jez6h1sfVIqyy_-Y985ZDPuaUYi",
            "Аліфатичні ациклічні": "1AnyrGIN8YHPM6vauaGQiDK4s4KooNoMV",
            "Аліфатичні гетероциклічні": "1OTk9EkMMkjjAjGR7DqntQ8yXfDlqbcmk",
            "Ароматичні гетероциклічні": "1h8YgqtJ6H3P8ba7WHFgEsmooJ3uN4uzK",
            "Ароматичні гомоциклічні": "115cnVxY3NQah10ZNS50A18KHWbHEg5Qj"
        },
        "RDKit": {
            "Всі": "1OiFvjUKqblW4hom7mP-4FLGbbrOrjCy8",
            "Аліфатичні ациклічні": "1q_MkbGKWPzOGZ08BnVDW82TuTZFpAIUj",
            "Аліфатичні гетероциклічні": "1-cQEhRWDHyLHpDVVNgwQAttpAVg_dqFk",
            "Ароматичні гетероциклічні": "18RVgyCmbs27Y3LsG2u98VkwYLyA1XTDZ",
            "Ароматичні гомоциклічні": "1jWK0IcVJmNLlhrLVRe4Cp1mgOtkHJcD5"
        }
    },
    "Boosting": {
        "Mordred": {
            "Всі": "1w2ItdAsDj033TGVRNTpqrHlR8a4SLhev",
            "Аліфатичні ациклічні": "1B5yeeidgex9srxieXKxkylag93Iy9iW0",
            "Аліфатичні гетероциклічні": "1W7cvJMf-s99NVAgYlFlto8b2jM4acNFI",
            "Ароматичні гетероциклічні": "1I7QwKc0pPdgUY9UxeHkZOnBYto_buMkD",
            "Ароматичні гомоциклічні": "1IbNUY3CzA9KitH6DmQeG76xCk2XeqP28"
        },
        "PaDEL": {
            "Всі": "1hc1houoqpU0vWtWf73VV9rXimZQBNJQ_",
            "Аліфатичні ациклічні": "1znSN_NQsxDBXBVZ5FQrzg6lMnVI56OAw",
            "Аліфатичні гетероциклічні": "1esmh0zAePZFUaOloWPo6c924W3BFKeGm",
            "Ароматичні гетероциклічні": "1qwDQPQPHwwOr_33cAbcCd3BHPFvbEJ2q",
            "Ароматичні гомоциклічні": "15dH_u0NR87HGbDyO4Z_tQ7V2Uu4_Mpt2"
        },
        "RDKit": {
            "Всі": "1N41_t9myC9RNL7IFT2SVZtsyZn9LeGVt",
            "Аліфатичні ациклічні": "1w80izq8_jcopv-CpS4-TfYldMFVGaRwa",
            "Аліфатичні гетероциклічні": "1AUA51utgoZo8tGdIxJvIfizzShPJDlMk",
            "Ароматичні гетероциклічні": "12IFWcGkv73uZ4O1026bYVuWTla40v1vb",
            "Ароматичні гомоциклічні": "19DNaPEr5eB4UJV2lFx4MpIm23YdaVIwy" }
    },
    "Neural Network": {
        "Mordred": {
            "Всі": "1PXrEvA2nEcFi04Rvt0zuUX7QvOVGgo3W",
            "Аліфатичні ациклічні": "1D1DnRzQWGPXP-M-mam1zl_RCGRSzvoA5",
            "Аліфатичні гетероциклічні": "1GU6Y1gLrP6tfLtBC0TwAQ7v9VfQNpZuN",
            "Ароматичні гетероциклічні": "1YnMs2d4IMFYvncvs-NxBCXQmjr7Glcc1",
            "Ароматичні гомоциклічні": "1WIHJOUAYtvziM-us76csGTkB7qfgpM4a"
        },
        "PaDEL": {
            "Всі": "1lDRDFOGMhH3-Ev9TlefBDQb3ztJrezWr",
            "Аліфатичні ациклічні": "1_TpWE3J7ywXWx2Hh5Q_dgd348TB6tUSr",
            "Аліфатичні гетероциклічні": "16PeB5VnL_cTc4gNu-5iJz3FCWbpE1xw3",
            "Ароматичні гетероциклічні": "1LcDp2PDWcBbI-kvM5di7f4RrZF9HF0Ys",
            "Ароматичні гомоциклічні": "100Igu2pheFUz5Pd466RypXuoTcCDjCXl"
        },
        "RDKit": {
            "Всі": "1ZNTp1-mNxfjF0PbcwUqSo_q5NuXjPYlG",
            "Аліфатичні ациклічні": "1419IL9oUZbyNG7oaFYiSLYH27qL6y0GN",
            "Аліфатичні гетероциклічні": "17QDCWH7vAeXdoYoSvGKsNO3d4bb8rY1a",
            "Ароматичні гетероциклічні": "1d1SUx2XeQWiPH9Fa5CLFOb61_m5fNELC",
            "Ароматичні гомоциклічні": "1GdcB1XGPIDRLoxoRSCCsmQ90NVMpumy3"
        }
    }
}

# автоматичні правила
auto_rules = {
    "Всі": ("Boosting", "Mordred"),
    "Аліфатичні ациклічні": ("Random Forest", "RDKit"),
    "Аліфатичні гетероциклічні": ("Neural Network", "PaDEL"),
    "Ароматичні гетероциклічні": ("Random Forest", "RDKit"),
    "Ароматичні гомоциклічні": ("Random Forest", "Mordred")
}


def load_model(model_type, descriptor, chem_class):
    # Беремо file_id із available_models2
    file_id = available_models2[model_type][descriptor][chem_class]

    # Беремо точну назву файлу з available_models
    filename = available_models[model_type][descriptor][chem_class]

    # Завантажуємо файл із Google Drive, якщо він ще не існує
    path = download_model_from_drive(file_id, filename)

    # Завантажуємо пайплайн
    return joblib.load(path)

def make_prediction(df, pipeline_bundle, model_type):
    """Обробка даних та прогноз"""

    if model_type in ["Random Forest", "Boosting"]:
        selector = pipeline_bundle.get("selector")
        scaler = pipeline_bundle.get("scaler")
        quantile_transformer = pipeline_bundle.get("quantile_transformer")
        all_features = pipeline_bundle["all_feature_names"]
        model = pipeline_bundle["model"]

        # Перевірка наявності всіх потрібних дескрипторів
        missing_cols = set(all_features) - set(df.columns)
        if missing_cols:
            st.error(f"❌ У файлі не вистачає дескрипторів для моделі: {missing_cols}")
            st.info("Скористайтеся сервісами для обчислення дескрипторів")
            st.stop()

        X = df[all_features]
        X = X.replace([np.inf, -np.inf], 0).fillna(0)
        X = X.astype(np.float64)
        if quantile_transformer is not None:
            X = quantile_transformer.transform(X)
        if scaler is not None:
            X = scaler.transform(X)
        if selector is not None:
            X = selector.transform(X)
        preds = model.predict(X)

    elif model_type == "Neural Network":
        model = pipeline_bundle["model"]
        selected_features = pipeline_bundle["selected_feature_names"]
        missing_cols = set(selected_features) - set(df.columns)
        if missing_cols:
            st.error(f"❌ У файлі не вистачає дескрипторів для моделі: {missing_cols}")
            st.info("Скористайтеся сервісами для обчислення дескрипторів")
            st.stop()
        X = df[selected_features]
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = np.float64(X.values)

        X = QuantileTransformer(output_distribution='uniform').fit_transform(X)
        X = StandardScaler().fit_transform(X)
        preds = model.predict(X)

    # додаємо результат
    df_out = df.copy()
    labels = ["Ризик мутагенності високий" if p >= 0.5 else "Ризик мутагенності низький" for p in preds]
    df_out["Prediction"] = labels

    return df_out

# --- Streamlit UI ---
st.title("🧬 AmesScan")
st.set_page_config( page_title="AmesScan", page_icon="🧬", layout="wide")
st.markdown(
    """
    **Інструкція для вхідного файлу:**  
    - Файл має містити стовпець із **SMILES-нотаціями хімічних сполук**, який може мати одну з таких назв: SMILES, Canonical SMILES або smiles.
    - Файл має містити стовпці з числовими молекулярними дескрипторами, причому перший рядок повинен визначати назви відповідних параметрів.
    - Якщо файл містить лише SMILES-нотації, можна автоматично розрахувати RDKit дескриптори для подальшого прогнозу.  
    """
)
st.sidebar.header("Налаштування")

descriptor_choice = st.sidebar.selectbox(
    "Оберіть набір дескрипторів",
    ["Автоматичні", "PaDEL", "RDKit", "Mordred"]
)

chem_class = st.sidebar.selectbox(
    "Оберіть клас",
    ["Всі", "Аліфатичні ациклічні", "Аліфатичні гетероциклічні", "Ароматичні гетероциклічні", "Ароматичні гомоциклічні"]
)

if descriptor_choice == "Автоматичні":
    # визначаємо модель і дескриптори з правил
    model_type, descriptor_choice = auto_rules[chem_class]
    st.sidebar.info(f"Автоматичний вибір: {model_type} + {descriptor_choice}")
else:
    model_type = st.sidebar.selectbox("Оберіть модель", list(available_models.keys()))

st.sidebar.markdown("### 🔗 Ресурси для обчислення дескрипторів:")
st.sidebar.markdown(
    """
    - [PaDEL](https://usegalaxy.eu/root?tool_id=padel&utm_source=chatgpt.com)
    - [Mordred](https://cheminformatics.usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fbgruening%2Fmordred%2Fctb_mordred_descriptors&utm_source=chatgpt.com)
    - [RDKit](https://usegalaxy.eu/root?tool_id=ctb_rdkit_descriptors&utm_source=chatgpt.com)
    """
)

uploaded_file = st.file_uploader("Завантажте файл (.csv або .xls)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Не вдалося прочитати файл: {e}")
        st.stop()

        # Перевірка на пустий файл
    if df.empty:
        st.warning("⚠️ Файл порожній. Завантажте файл з даними для прогнозу.")
        st.stop()

    st.write("📂 Вхідні дані:")
    st.dataframe(df.head(), use_container_width=True)

    # Перевіряємо, чи є дескриптори
    if set(df.columns) <= {"SMILES", "Canonical SMILES", "smiles"}:
        st.warning("Знайдено лише SMILES-нотації. Для прогнозу потрібні дескриптори.")

        if "computed_df" not in st.session_state:
            st.session_state.computed_df = None

        # --- Логіка показу кнопки ---
        show_compute_button = False
        if descriptor_choice == "RDKit":
            show_compute_button = True
        elif descriptor_choice == "Автоматичні" and auto_rules[chem_class][1] == "RDKit":
            show_compute_button = True

        if show_compute_button and st.button("Обчислити RDKit дескриптори автоматично"):
            st.info("Обчислення RDKit дескрипторів... будь ласка, зачекайте ⏳")

            # тут йде твій код обчислення дескрипторів RDKit
            rdkit_desc_names = [d[0] for d in Descriptors.descList]
            rdkit_data = []
            smiles_col = "SMILES" if "SMILES" in df.columns else "Canonical SMILES"
            invalid_smiles = []

            for i, smi in enumerate(df[smiles_col]):
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        values = [desc[1](mol) for desc in Descriptors.descList]
                        rdkit_data.append(values)
                    else:
                        invalid_smiles.append(smi)
                        rdkit_data.append([None] * len(rdkit_desc_names))
                except:
                    invalid_smiles.append(smi)
                    rdkit_data.append([None] * len(rdkit_desc_names))

            rdkit_df = pd.DataFrame(rdkit_data, columns=rdkit_desc_names)
            df = pd.concat([df, rdkit_df], axis=1)
            st.session_state.computed_df = df

            if invalid_smiles:
                st.warning(f"⚠️ Для {len(invalid_smiles)} молекул не вдалося обчислити дескриптори.")

            st.success("✅ RDKit дескриптори обчислено!")
            st.info("Тепер ви можете запустити прогноз.")


        # Якщо дескриптори вже обчислені — можна одразу запускати прогноз
        if st.session_state.computed_df is not None:
            df = st.session_state.computed_df

            if st.button("🚀 Запустити прогноз"):
                pipeline_bundle = load_model(model_type, descriptor_choice, chem_class)
                results = make_prediction(df, pipeline_bundle, model_type)

                st.success("✅ Прогноз виконано!")

                if "Canonical SMILES" in results.columns:
                    final_results = results[["Canonical SMILES", "Prediction"]]
                elif "SMILES" in results.columns:
                    final_results = results[["SMILES", "Prediction"]]
                else:
                    final_results = results[["Prediction"]]

                st.write("📝 Результати:")
                st.dataframe(final_results, use_container_width=True, height=400)

                output_file = "output.csv"
                final_results.to_csv(output_file, index=False)
                with open(output_file, "rb") as f:
                    st.download_button("⬇️ Завантажити результати", f, file_name="predictions.csv")

        else:
            st.info("""
                Для розрахунку дескрипторів ви можете скористатися сервісами:
                - [PaDEL](https://usegalaxy.eu/root?tool_id=padel&utm_source=chatgpt.com)
                - [Mordred](https://cheminformatics.usegalaxy.eu/root?tool_id=toolshed.g2.bx.psu.edu%2Frepos%2Fbgruening%2Fmordred%2Fctb_mordred_descriptors&utm_source=chatgpt.com)
                """)

    else:
        # Якщо у файлі вже є дескриптори
        st.success("✅ У файлі виявлено дескриптори! Можна одразу виконати прогноз.")

        if st.button("🚀 Запустити прогноз"):
            pipeline_bundle = load_model(model_type, descriptor_choice, chem_class)
            results = make_prediction(df, pipeline_bundle, model_type)

            st.success("✅ Прогноз виконано!")

            if "Canonical SMILES" in results.columns:
                final_results = results[["Canonical SMILES", "Prediction"]]
            elif "SMILES" in results.columns:
                final_results = results[["SMILES", "Prediction"]]
            else:
                final_results = results[["Prediction"]]

            st.write("📝 Результати:")
            st.dataframe(final_results, use_container_width=True, height=400)

            output_file = "output.csv"
            final_results.to_csv(output_file, index=False)
            with open(output_file, "rb") as f:
                st.download_button("⬇️ Завантажити результати", f, file_name="predictions.csv")