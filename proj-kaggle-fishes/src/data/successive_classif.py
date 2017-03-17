def open_dump(path, textfile):
    return pickle.load(open(os.path.join(path, textfile), 'rb'))


INTERIM = '../../data/interim'
PROCESSED = '../../data/processed'

alldescriptors = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'alldescriptors.txt')
ifeatures = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'ifeatures.txt')
sfeatures = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'sfeatures.txt')
labels = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'labels.txt')
filenames = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'filenames.txt')
labels = open_dump(os.path.join(INTERIM, 'train', 'crop'), 'labels.txt')
df_80 = open_dump(PROCESSED, 'df_80.txt')
features = np.hstack([sfeatures, ifeatures])

classes = ['ALB',
                'BET',
                'DOL',
                'LAG',
                'SHARK',
                'YFT',
                'OTHER',
                'NoF'
                ]

labelsbin = []
for index, item in enumerate(labels):
    if int(item) > 0:
        labelsbin.append(1)
    else:
        labelsbin.append(0)

X_train = []
y_train = []
X_val = []
y_val = []

df_80_base = df_80['img_file'].apply(os.path.basename).str.extract('(img_\d*)', expand=False)
p = re.compile('img_\d*')
for row in range(0, len(features)):
    if any(df_80_base == p.match(os.path.basename(filenames[row])).group(0)):
        X_train.append(features[row])
        y_train.append(labelsbin[row])
    else:
        X_val.append(features[row])
        y_val.append(labelsbin[row])

X_train = np.array(X_train)
X_val = np.array(X_val)

def RunOptClassif(preproc, classifier):
    scaler_class = Pipeline([('preproc', preproc),
                             ('classifier', classifier)])
    scaler_class.fit(X_train, y_train)
    y_true, y_pred = y_val, scaler_class.predict_proba(X_val)
    return str(classifier), log_loss(y_true, y_pred), scaler_class.predict(X_val)


classifier = xgb.XGBClassifier(learning_rate=0.07,
                               n_estimators=100,
                               max_depth=7,
                               min_child_weight=1,
                               gamma=0.6,
                               reg_alpha=0.1,
                               subsample=0.8,
                               colsample_bytree=0.8,
                               objective='binary:logistic',
                               nthread=6,
                               scale_pos_weight=1,
                               seed=70)
results = RunOptClassif(preproc=StandardScaler(),
                             classifier=classifier)
print("Used :%s" % results[0])
print("Logloss score on validation set : %s" % results[1])
cnf_matrix = confusion_matrix(y_val, results[2])
print(cnf_matrix)



labelsnoalb = []
featuresnoalb = []
filenamesnoalb = []
for index, item in enumerate(labels):
    if int(item) not in [0, 6, 7]:
        labelsnoalb.append(item)
        featuresnoalb.append(features[index])
        filenamesnoalb.append(filenames[index])


X_train = []
y_train = []
X_val = []
y_val = []

df_80_base = df_80['img_file'].apply(os.path.basename).str.extract('(img_\d*)', expand=False)
p = re.compile('img_\d*')
for row in range(0, len(featuresnoalb)):
    if any(df_80_base == p.match(os.path.basename(filenamesnoalb[row])).group(0)):
        X_train.append(featuresnoalb[row])
        y_train.append(labelsnoalb[row])
    else:
        X_val.append(featuresnoalb[row])
        y_val.append(labelsnoalb[row])

X_train = np.array(X_train)
X_val = np.array(X_val)

def RunOptClassif(preproc, classifier):
    scaler_class = Pipeline([('preproc', preproc),
                             ('classifier', classifier)])
    scaler_class.fit(X_train, y_train)
    y_true, y_pred = y_val, scaler_class.predict_proba(X_val)
    return str(classifier), log_loss(y_true, y_pred), scaler_class.predict(X_val)

classifier = RandomForestClassifier(n_estimators=50)
# classifier = xgb.XGBClassifier(learning_rate=0.07,
#                                n_estimators=20,
#                                max_depth=7,
#                                min_child_weight=1,
#                                gamma=0.8,
#                                reg_alpha=0.4,
#                                subsample=0.6,
#                                colsample_bytree=0.8,
#                                objective='multi:softmax',
#                                nthread=6,
#                                scale_pos_weight=1,
#                                seed=80)
results = RunOptClassif(preproc=StandardScaler(),
                             classifier=classifier)
print("Used :%s" % results[0])
print("Logloss score on validation set : %s" % results[1])
cnf_matrix = confusion_matrix(y_val, results[2])
print(cnf_matrix)
