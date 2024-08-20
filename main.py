from numpy import array
from seaborn import regplot
from matplotlib.pyplot import scatter, show
from typing import List, Union
from pandas import Series

class SimpleLinearRegression:
    """Simple Linear Regression model"""

    def __init__(self):
        """Initializes the model"""
        self.b0 = 0
        self.b1 = 0
        self.x = None
        self.y = None
        self.x_test = None
        self.predict_class = None
        self.train_status = False
        self.coef_ = [self.b0, self.b1]

    def __str__(self):
        return f'SimpleLinearRegression(b0={self.b0}, b1={self.b1})'
    
    def __repr__(self):
        return f'SimpleLinearRegression(b0={self.b0}, b1={self.b1})'
    
    def __eq__(self, other: object) -> bool:
        if (other.b1 == self.b1) and (other.b0 == self.b0):
            return True
        return False

    def show_equation(self, print_equation: bool = True) -> str:
        equation = f'y = {self.b0} + {self.b1}x'
        
        if print_equation:
            print(equation)

        return equation

    def fit(self, x: List[Union[list, tuple, array]], y: List[Union[list, tuple, array]]) -> None:
        """ Trains the model """
        if (type(x) != type(y)):
            raise ValueError("The type of x and y must be the same")
        
        if not (type(x) == Series) and not (type(y) == Series):
            x = Series(x)
            y = Series(y)

        if (len(x) == len(y)):
            self.x = x
            self.y = y
            B1 = (x.cov(y))/(x.var())
            B0 = y.mean() - B1 * x.mean()
            self.b1 = B1
            self.b0 = B0
            self.train_status = True
        else:
            raise ValueError("The number of elements in x and y must be the same")
        
    def predict(self, x_test: List[Union[list, tuple, array]]) -> array:
        """Predicts the value of y for a given array of x"""
        if not (self.train_status):
            raise ValueError("The model has not been trained")
        
        if len(x_test) == 0:
            raise ValueError("The number of elements in x_test must be greater than 0")
        
        y_pred = array([self.b0 + self.b1 * xi for xi in x_test])
        self.x_test = x_test
        self.predict_class = y_pred

        return y_pred
        
    def predict_value(self, x_test: List[Union[float, int]]) -> float:
        """Predicts the value of y for a unique given x"""
        if not (self.train_status):
            raise ValueError("The model has not been trained")
        
        if not isinstance(x_test, (int, float)):
            raise ValueError("The value of x_test must be an integer or float")
        
        y_pred = self.b0 + self.b1 * x_test

        return y_pred

    def square_error(self, class_pred: List[Union[list, tuple, array]], y_real: List[Union[list, tuple, array]]) -> float:
        """Returns the square error of the prediction"""
        if (type(class_pred) != type(y_real)):
            class_pred = array(class_pred)
            y_real = array(y_real)

        if (len(class_pred) != len(y_real)):
            raise ValueError("The number of elements in class_pred and y_real must be the same")
        
        size = (len(class_pred) + len(y_real))/2
        square_error_list = [(class_pred[i] - y_real[i])**2 for i in range(len(class_pred))]
        square_error = sum(square_error_list)/size

        return square_error
    
    def mean_square_error(self, class_pred: List[Union[list, tuple, array]], y_real: List[Union[list, tuple, array]]) -> float:
        """Returns the mean square error of the prediction"""
        if (type(class_pred) != type(y_real)):
            class_pred = array(class_pred)
            y_real = array(y_real)

        if (len(class_pred) != len(y_real)):
            raise ValueError("The number of elements in class_pred and y_real must be the same")
        
        mean = sum(y_real)/len(y_real)
        mean_square_error = self.square_error(class_pred, y_real)/mean

        return mean_square_error
        
    def r2_score(self, predict_class, y_test) -> float:
        """Returns the accuracy of the model"""
        if (len(predict_class) != len(y_test)):
            raise ValueError("The number of elements in x_test and y_test must be the same")

        if (not self.train_status):
            raise ValueError("The model has not been trained")
        
        y_mean = sum(y_test)/len(y_test)

        sqe = sum([(yi - y_mean)**2 for yi in predict_class])
        sqt = sum([(y_real - y_mean)**2 for y_real in y_test])
        r2 = sqe/sqt

        return r2
    
    def regression_plot(self) -> None:
        """Plots the regression line"""
        if not (self.train_status):
            raise ValueError("The model has not been trained")
    
        regplot(x=self.x, y=self.y, line_kws={'color': 'red', 'lw': 2})
        scatter(self.x_test, self.predict_class, color='yellow')
        show()
