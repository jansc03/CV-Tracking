import numpy as np
import cv2
"""Diese Klasse wird genutzt zum Vorhersagen der nächsten Position der Box. 
    Durch die Konfiguration des Kalman können zwei Werte vorhergesagt werden, welche bei uns x und y sind,
    allerings haben wir aufgrund von Problemen mit der Vorhersage der Y Werte, diese im späteren Verlauf nicht weiter genutzt """
class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    # Prozessrauschkovarianzmatrix Q - erhöhtes Vertrauen in die Messung
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-1  # Beispielwert für Q (kleiner Wert für hohe "Lernrate")

    # Messrauschkovarianzmatrix R - verringertes Vertrauen in die Messung (kleinerer Wert bedeutet mehr Vertrauen)
    kf.measurementNoiseCov = np.array([[1e-2, 0],  # Weniger Unsicherheit bei Messungen
                                       [0, 1e-2]], dtype=np.float32)  # Für x, y Messungen

    """Diese Methode gibt einmal die tatsächlichen Werte über, für den Kalmanfilter zum lernen und Predicted im anschluss die nächsten Werte"""
    def predict(self,coordX,coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = predicted[0], predicted[1]
        return x, y