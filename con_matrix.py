import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

actual=np.random.binomial(1,.9,size=1000)
predicted=np.random.binomial(1,0.9,size=1000)

confusion_matrix=metrics.confusion_matrix(actual,predicted)

cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[0,1])

print(f"Accuracy : {metrics.accuracy_score(actual,predicted)}\nPrecision : {metrics.precision_score(actual,predicted)}\nSensitivity : {metrics.recall_score(actual,predicted)}\nF1_Score : {metrics.f1_score(actual,predicted)}")

cm_display.plot()
plt.show()