vectorizer = CountVectorizer(ngram_range=(1,1), max_df=0.75, max_features= None)
traindatavecs = vectorizer.fit_transform(X_train)
testdatavecs = vectorizer.transform(X_test)

vectorizer_tf = TfidfVectorizer(ngram_range=(1,1), max_df=0.75, max_features= None, norm='l1', use_idf=True)
traindatavecs_tf = vectorizer_tf.fit_transform(X_train)
testdatavecs_tf = vectorizer_tf.transform(X_test)

sub_sg = SGDClassifier(alpha= 1e-05, n_iter= 10, penalty= 'elasticnet', random_state=42)
log_reg = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', n_jobs=-1, random_state=42) 
svm = LinearSVC( C=1, random_state=42)


print("Fitting data to LinearSVM Classifier - Count")
svm.fit(traindatavecs, y_train)
y_pred_sv = svm.predict(testdatavecs)
report_sv = classification_report(y_test, y_pred_sv)
print(report_sv)
acc_cnt = accuracy_score(y_test, y_pred_sv)*100 
print(acc_cnt)
print("Fitting data to LinearSVM Classifier - Tf-Idf")
svm.fit(traindatavecs_tf, y_train)
y_pred_sv_tf = svm.predict(testdatavecs_tf)
report_sv_tf = classification_report(y_test, y_pred_sv_tf)
print(report_sv_tf)
acc_sv_tf = accuracy_score(y_test, y_pred_sv_tf)*100 
print(acc_sv_tf)
print ()

print("Fitting data to SGD Classifier - Count")
sub_sg.fit(traindatavecs, y_train)
y_pred = sub_sg.predict(testdatavecs)
report = classification_report(y_test, y_pred)
print(report)
acc = accuracy_score(y_test, y_pred)*100 
print(acc)
print("Fitting data to SGD Classifier - TF-Idf")
sub_sg.fit(traindatavecs_tf, y_train)
y_pred_tf = sub_sg.predict(testdatavecs_tf)
report_tf = classification_report(y_test, y_pred_tf)
print(report_tf)
acc_tf = accuracy_score(y_test, y_pred_tf)*100 
print(acc_tf)
print ()

print("Fitting data to Logistic Regression Classifier - Count")
log_reg.fit(traindatavecs, y_train)
y_pred_log = log_reg.predict(testdatavecs)
report_log = classification_report(y_test, y_pred_log)
print(report_log)
acc_log = accuracy_score(y_test, y_pred_log)*100 
print(acc_log)
print("Fitting data to Logistic Regression Classifier - TF-Idf")
log_reg.fit(traindatavecs_tf, y_train)
y_pred_log_tf = log_reg.predict(testdatavecs_tf)
report_log_tf = classification_report(y_test, y_pred_log_tf)
print(report_log_tf)
acc_log_tf = accuracy_score(y_test, y_pred_log_tf)*100 
print(acc_log_tf)