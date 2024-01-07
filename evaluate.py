
y_true = [0, 1, 2, 2, 0]
y_pred = [0, 0, 2, 2, 0]
target_names = []
for i in range(36):
    target_names.append('class'+str(i))
# print(classification_report(y_true, y_pred, target_names=target_names))
print(target_names)