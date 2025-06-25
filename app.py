%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# --- Configuration de la page Streamlit ---
st.set_page_config(page_title="Détection de Fraude Bancaire", layout="wide", initial_sidebar_state="expanded")

# --- Titre de l'application ---
st.title("🛡️ Application de Détection de Fraude Bancaire")
st.markdown("""
    Cette application permet de prédire la probabilité de fraude pour des transactions bancaires.
    Veuillez entrer les détails de la transaction ci-dessous.
""")

# --- Chargement des modèles et préprocesseurs ---
@st.cache_data # Mise en cache pour éviter de recharger à chaque interaction
def load_assets():
    try:
        base_path = os.path.dirname(__file__)
        
        model_path = os.path.join(base_path, 'best_logistic_regression_model.pkl')
        scaler_path = os.path.join(base_path, 'scaler.pkl')
        ohe_path = os.path.join(base_path, 'one_hot_encoder.pkl')
        imputer_num_path = os.path.join(base_path, 'imputer_numeric.pkl')
        imputer_cat_path = os.path.join(base_path, 'imputer_categorical.pkl')
        feature_cols_path = os.path.join(base_path, 'feature_columns.pkl')

        best_logistic_regression_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        one_hot_encoder = joblib.load(ohe_path)
        imputer_numeric = joblib.load(imputer_num_path)
        imputer_categorical = joblib.load(imputer_cat_path)
        feature_columns = joblib.load(feature_cols_path)

        st.sidebar.success("Modèle et préprocesseurs chargés avec succès !")
        return best_logistic_regression_model, scaler, one_hot_encoder, imputer_numeric, imputer_categorical, feature_columns
    except FileNotFoundError as e:
        st.error(f"Erreur: Fichier manquant. Assurez-vous que tous les fichiers .pkl sont dans le même répertoire que app.py. Détail: {e}")
        st.info(f"Fichiers attendus: best_logistic_regression_model.pkl, scaler.pkl, one_hot_encoder.pkl, imputer_numeric.pkl, imputer_categorical.pkl, feature_columns.pkl")
        st.stop()
    except Exception as e:
        st.error(f"Une erreur inattendue s'est produite lors du chargement des actifs: {e}")
        st.stop()

best_logistic_regression_model, scaler, one_hot_encoder, imputer_numeric, imputer_categorical, feature_columns = load_assets()

# --- Définition des colonnes numériques et catégorielles pour l'input ---
numerical_cols = [
    'Amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'
]
categorical_cols = [
    'Type'
]

# --- Interface utilisateur pour l'entrée des données ---
st.header("Saisie des détails de la transaction")

# Input pour la colonne 'Type' (catégorielle)
type_options = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN']
selected_type = st.selectbox("Type de transaction", type_options)

col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Montant de la transaction", min_value=0.0, format="%.2f")
    oldbalanceOrg = st.number_input("Solde initial du compte d'origine", min_value=0.0, format="%.2f")
with col2:
    newbalanceOrig = st.number_input("Nouveau solde du compte d'origine", min_value=0.0, format="%.2f")
    oldbalanceDest = st.number_input("Solde initial du compte de destination", min_value=0.0, format="%.2f")
with col3:
    newbalanceDest = st.number_input("Nouveau solde du compte de destination", min_value=0.0, format="%.2f")

# Prédiction
if st.button("Prédire la fraude"):
    # Création du DataFrame d'entrée
    input_data = pd.DataFrame({
        'Type': [selected_type],
        'Amount': [amount],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # Imputation des valeurs manquantes (si le Imputer a des stratégies spécifiques)
    input_numerical = input_data[numerical_cols]
    input_categorical = input_data[categorical_cols]

    # --- MODIFICATION ICI POUR L'IMPUTATION NUMÉRIQUE ---
    if imputer_numeric:
        # Passer les VALEURS (array numpy) à transform pour éviter le problème de noms de colonnes
        input_numerical_imputed_array = imputer_numeric.transform(input_numerical.values)
        # Recréer le DataFrame avec les bons noms de colonnes et l'index
        input_numerical_imputed = pd.DataFrame(input_numerical_imputed_array, columns=numerical_cols, index=input_numerical.index)
    else:
        input_numerical_imputed = input_numerical # Pas d'imputation numérique si imputer_numeric est None
    
    if imputer_categorical:
        input_categorical_imputed = pd.DataFrame(imputer_categorical.transform(input_categorical), columns=categorical_cols, index=input_categorical.index)
    else:
        input_categorical_imputed = input_categorical


    # One-Hot Encoding
    try:
        input_encoded = one_hot_encoder.transform(input_categorical_imputed)
        input_encoded_df = pd.DataFrame(input_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_cols), index=input_categorical_imputed.index)
    except Exception as e:
        st.error(f"Erreur lors de l'encodage One-Hot. Assurez-vous que toutes les catégories possibles sont gérées : {e}")
        st.stop()


    # Concaténation des caractéristiques numériques et encodées
    processed_data = pd.concat([input_numerical_imputed, input_encoded_df], axis=1)

    # Réordonnancement et alignement des colonnes avec feature_columns (très important)
    final_input = pd.DataFrame(columns=feature_columns)
    for col in feature_columns:
        if col in processed_data.columns:
            final_input[col] = processed_data[col]
        else:
            final_input[col] = 0 # Remplir avec 0 pour les colonnes manquantes (si une catégorie n'est pas présente)
    
    final_input = final_input.fillna(0)

    # --- MODIFICATION ICI POUR LE SCALING ---
    final_input_scaled = final_input.copy()
    # Passer les VALEURS (array numpy) à transform pour le scaler aussi
    # Appliquer le scaler uniquement aux colonnes numériques qui étaient utilisées pour entraîner le scaler
    final_input_scaled[numerical_cols] = scaler.transform(final_input_scaled[numerical_cols].values)

    # Prédiction de la probabilité
    prediction_proba = best_logistic_regression_model.predict_proba(final_input_scaled)[:, 1][0]
    
    # Interprétation du résultat
    st.subheader("Résultat de la prédiction :")
    if prediction_proba >= 0.5: # Seuil de 0.5 pour la détection de fraude
        st.error(f"**Fraude détectée !** Probabilité : {prediction_proba:.2%}")
        st.warning("Cette transaction présente une forte probabilité de fraude. Une vérification est recommandée.")
    else:
        st.success(f"**Non-fraude détectée.** Probabilité : {prediction_proba:.2%}")
        st.info("Cette transaction semble légitime.")

    st.markdown("---")
    st.subheader("Détails des données d'entrée traitées :")
    st.dataframe(final_input_scaled)

# --- Section d'information supplémentaire (Sidebar) ---
st.sidebar.header("À propos de l'application")
st.sidebar.info("""
    Cette application utilise un modèle de régression logistique entraîné pour classifier
    les transactions bancaires comme frauduleuses ou non.
""")
st.sidebar.markdown("---")
st.sidebar.write("Développé par [Votre Nom/Organisation]")
st.sidebar.write("Version : 1.0")

# --- Visualisation des métriques (si vous voulez ajouter des images statiques) ---
try:
    img_dir = os.path.dirname(__file__)
    st.image(os.path.join(img_dir, "ROC.png"), caption="Courbe ROC", use_column_width=True)
    st.image(os.path.join(img_dir, "confusion.png"), caption="Matrice de Confusion", use_column_width=True)
except FileNotFoundError:
    st.warning("Les images de performance (ROC.png, confusion.png) n'ont pas été trouvées. Assurez-vous de les uploader sur GitHub.")


