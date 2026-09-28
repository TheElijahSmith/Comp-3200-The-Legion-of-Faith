import numpy as np

from part1_refactored_loop import train
from part3_build_from_diagram import train_from_diagram

## Test 1
def test_refactor_correctness():
 tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])
  
  strike = np.array([1, 1, 0, 0])
  
  history = train(
    tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1)
  
  expected = 0.0000150556
  
  np.testing.assert_allclose(
    history[-1], expected, atol=1e-9)

## Test 2
def test_shape_sanity():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])
  
  strike = np.array([1, 1, 0, 0])
  
  hidden_size = 4
  
  np.random.seed(1)
  
  weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
  weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
  
  layer_0 = tells[0:1]
  goal = strike[0:1].reshape(1, 1)
  
  layer_1 = np.maximum(0, layer_0 @ weights_0_1)
  layer_2 = layer_1 @ weights_1_2
  
  layer_2_delta = layer_2 - goal
  layer_1_delta = (
  layer_2_delta @ weights_1_2.T
  ) * (layer_1 > 0).astype(float)
  
  assert layer_0.shape == (1, 3)
  assert layer_1.shape == (1, 4)
  assert layer_2.shape == (1, 1)
  
  assert layer_1_delta.shape == (1, 4)
  assert layer_2_delta.shape == (1, 1)
  assert weights_0_1.shape == (3, 4)
  assert weights_1_2.shape == (4, 1)

## Test 3
def test_deeper_net_runs():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])
  
  strike = np.array([1, 1, 0, 0])
  
  epochs = 10
  
  history = train_from_diagram( tells, strike, alpha=0.1, epochs=epochs, seed=4)
  
  assert len(history) == epochs

## Test 4
def test_deeper_net_converges():
 tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])
  
  strike = np.array([1, 1, 0, 0])
  
  alpha = 0.1
  epochs = 150
  seed = 4
  
  history = train_from_diagram(tells, strike, alpha=alpha, epochs=epochs, seed=seed)
  
  assert history[-1] < 1e-3

## Test 5
def test_determinism():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
])
  
  strike = np.array([1, 1, 0, 0])
  
  history1 = train_from_diagram(tells, strike, alpha=0.1, epochs=150, seed=4)
  
  history2 = train_from_diagram(tells, strike, alpha=0.1, epochs=150, seed=4)
  
  np.testing.assert_allclose(history1, history2)

## Main
if __name__ == "__main__":
  tests = [
    value
    for name, value in globals().items()
    if name.startswith("test_") and callable(value)
  ]
  for test in tests:
    try: 
      test()
      print(f"PASS: {test.__name__}")
    except AssertionError as e:
      print(f"FAIL: {test.__name__}")
      print(e)
