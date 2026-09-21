import numpy as np

from part1_single_layer_fails import single_layer_train
from part2_forward_hidden import relu, forward
from part3_one_backprop_step import relu2deriv, one_step
from part4_full_training import train

## ----- Test 1 --------
def test_relu():
  result = relu(np.array([-1, 0, 1, 2]))
  expected = np.array([0, 0, 1, 2])
  
  assert np.array_equal(result, expected)



## ----- Test 2 ------
def test_relu2deriv():
  result = relu2deriv(np.array([-1, 0, 1, 2]))
  expected = np.array([0, 0, 1, 1])
  
  assert np.array_equal(result, expected)

## ----- Test 3 ------
def test_single_layer_failure():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  strike = np.array([1, 1, 0, 0])
  
  error = single_layer_train(tells, strike, 0.1, 60, 1)
  
  assert error > 0.5

## ----- Test 4 ------
def test_forward_layer1_shape():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  
  np.random.seed(1)
  hidden_size = 4
  
  weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
  weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
  
  layer_1, layer_2 = forward(
    tells[0:1],
    weights_0_1,
    weights_1_2
  )
  
  assert layer_1.shape == (1, hidden_size)

## ---- Test 5 -------
def test_forward_layer2_shape():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  np.random.seed(1)
  hidden_size = 4
  
  weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
  weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
  
  layer_1, layer_2 = forward(
    tells[0:1],
    weights_0_1,
    weights_1_2
  )
  
  assert layer_2.shape == (1, 1)

## ---- Test 6 ------
def test_one_step_reduce_error():
  tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  
  strike = np.array([
    [1],
    [1],
    [0],
    [0]
  ])
  
  np.random.seed(1)
  hidden_size = 4
  
  weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
  weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
  
  layer_0 = tells[0:1]
  target = strike[0:1]
  
  layer_1_before = relu(layer_0.dot(weights_0_1))
  layer_2_before = layer_1_before.dot(weights_1_2)
  
  error_before = np.sum((layer_2_before - target) ** 2)
  
  update_w01, update_w12, _, _ = one_step(
    layer_0, target, weights_0_1, weights_1_2, 0.2
  )
  
  layer_1_after = relu(layer_0.dot(updated_w01))
  layer_2_after = layer_1_after.dot(updated_w12)
  
  error_after = np.sum((layer_2_after - target) ** 2)
  
  assert error_after < error_before

## ----- Test 7 ------
def test_full_training_convergence():
   tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  
  strike = np.array([
    [1],
    [1],
    [0],
    [0]
  ])

weights_0_1, weights_1_2, error_history = train(
  tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1
)

assert error_history[-1] < 1e-2

for i in range(len(tells)):
  layer_0 = tells[i:i+1]
  layer_1 = relu(layer_0.dot(weights_0_1))
  layer_2 = layer_1.dot(weights_1_2)
  
  prediction = layer_2[0,0]
  
  if strike[i, 0] == 1:
    assert prediction > 0.5
  else:
    assert prediction < 0.5

## ----- Test 8 -----
def test_determinism():
   tells = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1]
  ])
  
  strike = np.array([
    [1],
    [1],
    [0],
    [0]
  ])

weights_0_1_a, weights_1_2_a, _ = train(
  tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1
)

weights_0_1_b, weights_1_2_b, _ = train(
 tells, strike, alpha=0.2, epochs=60, hidden_size=4, seed=1
)

assert np.allclose(weights_0_1_a, weights_0_1_b, atol=1e-10
assert np.allclose(weights_1_2_a, weights_1_2_b, atol=1e-10


if __name__ == "__main__":
    for name, test in globals().items():
      if name.startswith("test_") and callable(test):
        try:
          test()
          print(f"PASS: {name}")
        except Exception as e:
          print(f"FAIL: {name} -> {e}")

  
