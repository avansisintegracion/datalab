from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler, SMOTE


# Apply the random over-sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(test.X_train, test.y_train)

RANDOM_STATE = 42
classifier = classifier = xgb.XGBClassifier(learning_rate=0.05,
                               n_estimators=100,
                               max_depth=7,
                               min_child_weight=1,
                               gamma=0.6,
                               reg_alpha=0.4,
                               subsample=0.6,
                               colsample_bytree=0.8,
                               objective='multi:softmax',
                               nthread=6,
                               scale_pos_weight=1,
                               seed=70)
pipeline = make_pipeline(NearMiss(version=3, random_state=RANDOM_STATE),
                         StandardScaler(),
                         classifier)
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix with NearMiss v2')
plt.savefig(os.path.join(INTERIM, 'XGBoost_NearMiss_confusion_matrix.png'),
            bbox_inches='tight')


pipeline = make_pipeline(StandardScaler(),
                         classifier)
pipeline.fit(test.X_train, test.y_train)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
print(log_loss(y_true, y_pred))

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix')
plt.savefig(os.path.join(INTERIM, 'XGBoost_confusion_matrix.png'),
            bbox_inches='tight')





from imblearn.over_sampling import RandomOverSampler


# Apply the random over-sampling
ros = RandomOverSampler()
X_resampled, y_resampled = ros.fit_sample(test.X_train, test.y_train)

pipeline = make_pipeline(StandardScaler(),
                         classifier)
pipeline.fit(X_resampled, y_resampled)
cnf_matrix = confusion_matrix(test.y_val, pipeline.predict(test.X_val))
print(cnf_matrix)
print(classification_report_imbalanced(test.y_val, pipeline.predict(test.X_val), target_names=test.classes))
y_true, y_pred = test.y_val, pipeline.predict_proba(test.X_val)
log_loss(y_true, y_pred)

np.set_printoptions(precision=2)
# Plot normalized confusion matrix
plt.figure()
test.plot_confusion_matrix(cnf_matrix,
                           classes=test.classes,
                           normalize=False,
                           title='Confusion matrix with RandomSampleOver')
plt.savefig(os.path.join(INTERIM, 'XGBoost_RSO_confusion_matrix.png'),
            bbox_inches='tight')
