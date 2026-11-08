from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp('2025-10-01 00:00:00')
CUTOFF = pd.Timestamp('2025-09-01 00:00:00')


def load_data():
    orders = pd.read_csv(DATA / 'clean' / 'orders.csv', parse_dates=['order_ts'])
    customers = pd.read_csv(DATA / 'clean' / 'customers.csv', parse_dates=['signup_date'])
    products = pd.read_csv(DATA / 'clean' / 'products.csv')
    products['price_inr'] = pd.to_numeric(products['price_inr'])
    orders = orders.merge(products[['product_id', 'category']], on='product_id', how='left', validate='many_to_one')
    return orders, customers, products


def build_rfm(orders, customers):
    completed = orders.loc[orders.status.eq('completed')].copy()
    rfm = completed.groupby('customer_id').agg(
        last_order_ts=('order_ts', 'max'),
        frequency=('order_id', 'nunique'),
        monetary=('order_amount', 'sum')
    )
    rfm['recency'] = (SNAPSHOT - rfm['last_order_ts']).dt.days
    rfm = rfm[['recency', 'frequency', 'monetary']].reset_index()
    print('\nRFM SUMMARY\n', rfm[['recency','frequency','monetary']].agg(['count','mean','median','min','max']).round(2))
    assert len(rfm) == 1175
    assert int(rfm.recency.median()) == 19 and int(rfm.recency.min()) == 0 and int(rfm.recency.max()) == 180
    assert round(rfm.frequency.mean(), 2) == 9.46 and int(rfm.frequency.max()) == 33
    assert round(rfm.monetary.mean(), 2) == 2552.06 and round(rfm.monetary.max(), 2) == 14766.55
    return rfm


def cluster_customers(rfm, customers):
    cols = ['recency','frequency','monetary']
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[cols])
    rows = []
    models = {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        models[k] = km
        rows.append({'K':k, 'inertia':km.inertia_, 'silhouette':silhouette_score(X, labels)})
    metrics = pd.DataFrame(rows)
    print('\nK-MEANS METRICS\n', metrics.round(3).to_string(index=False))
    # K=4 is selected: clear elbow and a business-actionable four-segment profile.
    chosen_k = 4
    labels = models[chosen_k].labels_
    clustered = rfm.copy()
    clustered['segment_id'] = labels
    profiles = clustered.groupby('segment_id').agg(
        size=('customer_id','size'), mean_recency=('recency','mean'),
        mean_frequency=('frequency','mean'), mean_monetary=('monetary','mean')
    ).sort_values('mean_monetary', ascending=False)
    premium_id = int(profiles.index[0])
    # Assign human-readable names based on each segment's observed RFM profile.
    names = {}
    for sid, p in profiles.iterrows():
        if sid == premium_id:
            names[int(sid)] = 'Premium Loyalists'
        elif p.mean_frequency >= profiles.mean_frequency.median() and p.mean_recency <= profiles.mean_recency.median():
            names[int(sid)] = 'Frequent Regulars'
        elif p.mean_recency > 60:
            names[int(sid)] = 'Dormant Customers'
        else:
            names[int(sid)] = 'Occasional Shoppers'
    clustered['segment_name'] = clustered.segment_id.map(names)
    clustered['is_premium'] = clustered.segment_id.eq(premium_id)
    print('\nSEGMENT PROFILES\n', profiles.round(2).to_string())
    print('Segment names:', names, '| Premium segment:', premium_id)
    pca = PCA(n_components=2)
    pca.fit(X)
    print('PCA explained variance ratio:', pca.explained_variance_ratio_, 'total:', pca.explained_variance_ratio_.sum())
    result = clustered.merge(customers[['customer_id']], on='customer_id', how='left')
    result = result[['customer_id','recency','frequency','monetary','segment_id','segment_name','is_premium']]
    result.to_csv(OUT / 'customer_segments.csv', index=False)
    return result, profiles, premium_id, names


def build_lapse_dataset(orders, customers):
    pre = orders.loc[orders.order_ts < CUTOFF].copy()
    completed_pre = pre.loc[pre.status.eq('completed')].copy()
    cohort_ids = completed_pre.customer_id.unique()
    post_window = orders.loc[(orders.order_ts >= CUTOFF) & (orders.order_ts < pd.Timestamp('2025-10-01')) & orders.status.eq('completed')]
    active_ids = set(post_window.customer_id)
    labels = pd.DataFrame({'customer_id': cohort_ids})
    labels['lapsed'] = (~labels.customer_id.isin(active_ids)).astype(int)

    g = completed_pre.groupby('customer_id')
    features = g.agg(
        last_order_ts=('order_ts','max'), frequency=('order_id','nunique'),
        monetary=('order_amount','sum'), avg_discount_pct=('discount_pct','mean'),
        avg_delivery_minutes=('delivery_minutes','mean')
    )
    features['recency'] = (CUTOFF - features.pop('last_order_ts')).dt.days
    all_final = pre.loc[pre.status.isin(['completed','cancelled'])].copy()
    cancel = all_final.assign(is_cancelled=all_final.status.eq('cancelled').astype(int)).groupby('customer_id').agg(
        cancellation_rate=('is_cancelled','mean'))
    fav = completed_pre.groupby('customer_id')['category'].agg(lambda s: s.mode().sort_values().iloc[0])
    features = features.join(cancel).join(fav.rename('favourite_category'))
    features = features.reset_index().merge(customers, on='customer_id', how='left')
    features['tenure_days'] = (CUTOFF - features['signup_date']).dt.days
    features = features.drop(columns=['signup_date'])
    dataset = features.merge(labels, on='customer_id', validate='one_to_one')
    # The model sees only pre-cutoff behavioral features and customer profile attributes.
    feature_cols = ['recency','frequency','monetary','avg_discount_pct','cancellation_rate',
                    'avg_delivery_minutes','favourite_category','city','age','gender','tenure_days']
    X = dataset[feature_cols].copy()
    y = dataset['lapsed'].astype(int)
    assert not X.isna().any().any(), 'Missing values in model features'
    print('\nLAPSE COHORT:', len(dataset), '| lapsed:', int(y.sum()), f'({y.mean()*100:.2f}%)')
    print('Feature missing values:', int(X.isna().sum().sum()))
    return dataset, X, y, feature_cols


def train_models(dataset, X, y, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    print('Train/test:', len(X_train), len(X_test))
    baseline = np.zeros(len(y_test), dtype=int)
    print('Baseline accuracy:', accuracy_score(y_test, baseline),
          'lapsed F1:', f1_score(y_test, baseline, zero_division=0))
    numeric = ['recency','frequency','monetary','avg_discount_pct','cancellation_rate','avg_delivery_minutes','age','tenure_days']
    categorical = ['favourite_category','city','gender']
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])
    log_pipe = Pipeline([('prep', clone(preprocessor)), ('model', LogisticRegression(max_iter=1000))])
    log_grid = GridSearchCV(log_pipe, {'model__C':[0.01,0.1,1,10,100]}, cv=5, scoring='f1', n_jobs=-1)
    log_grid.fit(X_train, y_train)
    fitted_scaler = log_grid.best_estimator_.named_steps['prep'].named_transformers_['num']
    freq_position = numeric.index('frequency')
    print('Frequency scaler mean (training rows):', fitted_scaler.mean_[freq_position],
          '| full-cohort frequency mean:', X['frequency'].mean())
    print('The means differ because the scaler is fitted only on the training split, while the full-cohort mean includes held-out test rows.')
    rf_pipe = Pipeline([('prep', clone(preprocessor)), ('model', RandomForestClassifier(n_estimators=100, random_state=42))])
    rf_grid = GridSearchCV(rf_pipe, {'model__max_depth':[3,5,8], 'model__min_samples_leaf':[5,10,20]}, cv=5, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    print('\nLOGISTIC best:', log_grid.best_params_, 'CV F1:', log_grid.best_score_)
    print('RANDOM FOREST best:', rf_grid.best_params_, 'CV F1:', rf_grid.best_score_)
    results = {}
    for name, search in [('logistic',log_grid),('random_forest',rf_grid)]:
        pred = search.best_estimator_.predict(X_test)
        cm = confusion_matrix(y_test, pred, labels=[0,1])
        metrics = {'confusion_matrix':cm.tolist(), 'precision':precision_score(y_test,pred,zero_division=0),
                   'recall':recall_score(y_test,pred,zero_division=0), 'f1':f1_score(y_test,pred,zero_division=0)}
        results[name] = metrics
        print(name, json.dumps(metrics))
    selected_name, selected_search = max([('logistic',log_grid),('random_forest',rf_grid)], key=lambda z: results[z[0]]['f1'])
    selected_model = selected_search.best_estimator_
    if results[selected_name]['f1'] < 0.55:
        raise AssertionError(f"Selected test F1 below 0.55: {selected_name} {results[selected_name]['f1']:.3f}")
    model_path = OUT / 'lapse_model.joblib'
    joblib.dump(selected_model, model_path)
    loaded = joblib.load(model_path)
    assert np.array_equal(selected_model.predict(X_test), loaded.predict(X_test))
    print('Selected model:', selected_name, '| reloaded predictions identical: True')

    # Training-only random oversampling for the best logistic regression.
    best_log = log_grid.best_estimator_
    train_frame = X_train.copy()
    train_frame['_y'] = y_train.values
    counts = train_frame['_y'].value_counts()
    majority = int(counts.max())
    minority_rows = train_frame.loc[train_frame['_y'].eq(1)]
    extra = minority_rows.sample(n=majority-int(counts.get(1,0)), replace=True, random_state=42)
    balanced = pd.concat([train_frame, extra], ignore_index=True).sample(frac=1, random_state=42)
    balanced_X, balanced_y = balanced.drop(columns='_y'), balanced['_y']
    balanced_log = clone(best_log).fit(balanced_X, balanced_y)
    orig_pred = best_log.predict(X_test)
    bal_pred = balanced_log.predict(X_test)
    print('Oversampling class counts:', balanced_y.value_counts().sort_index().to_dict())
    print('Logistic original P/R/F1:', precision_score(y_test,orig_pred,zero_division=0), recall_score(y_test,orig_pred,zero_division=0), f1_score(y_test,orig_pred,zero_division=0))
    print('Logistic balanced P/R/F1:', precision_score(y_test,bal_pred,zero_division=0), recall_score(y_test,bal_pred,zero_division=0), f1_score(y_test,bal_pred,zero_division=0))

    # Interpret model drivers.
    rf = rf_grid.best_estimator_
    feature_names = rf.named_steps['prep'].get_feature_names_out()
    importances = pd.Series(rf.named_steps['model'].feature_importances_, index=feature_names).sort_values(ascending=False)
    print('\nTOP RANDOM FOREST FEATURES\n', importances.head(5).to_string())
    lr = log_grid.best_estimator_
    lr_names = lr.named_steps['prep'].get_feature_names_out()
    coefs = pd.Series(lr.named_steps['model'].coef_[0], index=lr_names).sort_values(key=abs, ascending=False)
    print('\nLOGISTIC COEFFICIENTS (top 3 absolute)\n', coefs.head(3).to_string())

    # Illustrative post-hoc Premium list; premium flag is based on whole-window RFM.
    segments = pd.read_csv(OUT / 'customer_segments.csv')
    test_ids = X_test.index
    # Preserve customer_id mapping by row position through the original dataset index.
    test_customers = dataset.loc[test_ids, ['customer_id','lapsed']].copy()
    test_customers['_test_index'] = test_customers.index
    premium = test_customers.merge(segments[['customer_id','is_premium']], on='customer_id', how='left')
    premium = premium.loc[premium.is_premium.fillna(False)].copy()
    if len(premium):
        probabilities = selected_model.predict_proba(X_test.loc[premium['_test_index']])[:, list(selected_model.classes_).index(1)]
        premium['lapse_probability'] = probabilities
        premium['actually_lapsed'] = premium['lapsed'].astype(int)
        retention = premium.sort_values('lapse_probability', ascending=False).head(10)[['customer_id','lapse_probability','actually_lapsed']]
    else:
        retention = pd.DataFrame(columns=['customer_id','lapse_probability','actually_lapsed'])
    retention.to_csv(OUT / 'retention_list.csv', index=False)
    print('Premium test customers:', len(premium), '| retention rows:', len(retention), '| truly lapsed:', int(retention.actually_lapsed.sum()) if len(retention) else 0)


def main():
    orders, customers, products = load_data()
    rfm = build_rfm(orders, customers)
    segments, profiles, premium_id, names = cluster_customers(rfm, customers)
    dataset, X, y, feature_cols = build_lapse_dataset(orders, customers)
    train_models(dataset, X, y, feature_cols)
    print('\nPART B COMPLETE')

if __name__ == '__main__':
    main()
