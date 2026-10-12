import numpy as np
from data import load_mnist, one_hot
from part1_memorize import train
from part2_early_stopping import train_early_stopping
from part3_dropout import train_dropout
from part3_dropout import dropout_step
from part4_batching import train_batched


# Test 1
def test_overfitting():
  (X_all, y_all), _ = load_mnist()
  
  X_val = X_all[1000:2000]
  Y_val = one_hot(y_all[1000:2000])
  
  X_train = X_all[:1000]
  Y_train = one_hot(y_all[:1000])
  
  W_0_1, W_1_2, train_hist, val_hist = train(X_train, Y_train, X_val, y_val, iterations=60)
  
  final_train = train_hist[-1]
  final_val = val_hist[-1]
  gap = final_train - final_val
  
  assert final_train > 0.95, (f"Training accuracy too low: {final_train:.3f}")
  
  assert gap >= 0.10, (f"Overfitting gap too small: {gap:.3f}")
  


# Test 2
def test_no_test_set_leakage():
  training_functions = [train, train_early_stopping, train_dropout, train_batched,]
  
  for function in training_functions:
    parameters = inspect.signature(function).parameters
    
    for name in parameters:
      assert "test" not in name.lower(), (f"{function.__name__}has a test-related parameter: {name}")
      
      source = inspect.getsource(function)
      
      assert "X_test" not in source, (f"{function.__name__} mentions X_test")
      
      assert "y_test" not in source, (f"{functions.__name__} mentions y_test")



# Test 3
def test_early_stopping_restores_best():
  (X_all, y_all), _ = load_mnist()
  
  X_val = X_all[1000:2000]
  Y_val = one_hot(y_all[1000:2000])
  X_train = X_all[:1000]
  Y_train = one_hot(y_all[:1000])
  
  W_0_1, W_1_2, train_hist, val_hist = train_early_stopping(X_train, Y_train, X_val, Y_val, iterations=60, patience=10)
  
  best_val = max(val_hist)
  returned_val = train_early_stopping.__globals__["evaluate"](X_val, Y_val, W_0_1, W_1_2)
  
  assert abs(returned_val - best_val) < 1e-9, (
    f"Returned weights give validation accuracy {returned_val:.6f}, "
    f"but best recorded accuracy was {best_val:.6f}")




# Test 4
def test_dropout_narrows_gap():
  (X_all, y_all), _ = load_mnist()
  
  X_val = X_all[1000:2000]
  Y_val = one_hot(y_all[1000:2000])
  X_train = X_all[:1000]
  Y_train = one_hot(y_all[:1000])
  
  _, _, train_no_dropout, val_no_dropout = train_dropout(X_train, Y_train, X_val, Y_val,iterations=10, dropout=False)
  
  _, _, train_dropout_hist, val_dropout_hist = train_dropout(X_train, Y_train, X_val, Y_val, iterations=10, dropout=True)
  
  gap_no_dropout = train_no_dropout[01] - val_no_dropout[-1]
  gap_dropout = train_dropout_hist[-1] - val_dropout_hist[-1]
  
  assert gap_dropout < gap_no_dropout, (f"Dropout gap({gap_dropout:.3f}) was not smaller than "
                                        f"the no-dropout gap({gap_no_dropout:.3f})")



# Test 5
def test_dropout_gradient():
  layer_0 = np.array([1.0, 2.0])
  
  target = np.array([1.0, 0.0])
  
  W_0_1 = np.array([
    [0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7]
  ])
  
  w_1_2 = np.arrau([
    [0.2, 0.3],
    [0.4, 0.5],
    [0.6, 0.7]
  ])
  
  mask = np.array([[1, 0, 1]])
  
  original_W_0_1 = w_0_1.copy()
  
  dropout_step(layer_0, target, w_0_1, W_1_2, mask, alpha=0.01)
  
  np.testing.assert_array_equal(w_0_1[:, 1], original_w_0_1[:, 1])
  
  assert not np.array_equal(W_0_1[:, 0], original_w_0_1[:, 0]),
  
  assert not np.array_equal(W_0_1[:, 2], original_w_0_1[:, 2]),
  




# Testing the functions
def main():
  tests = [test_overfitting, test_no_set_leakage, test_early_stopping_restores_best, test_dropout_narrows_gap, test_dropout_gradient,]
  for test in tests:
    try:
         test()
         print(f"PASS: {test.__name__}")
    except Exception as e:
         print(f"FAIL: {test.__name__}: {e}")

if __name__ == "__main__":
  main()
      


    
          
           







  



