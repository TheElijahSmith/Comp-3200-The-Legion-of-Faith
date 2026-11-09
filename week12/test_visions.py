import numpy as np
from preprocesses import vocab, word2index, words2indices, tokens
from rnn_forward import init_weights, softmax, predict, compute_perplexity
from rnn_bptt import train_step, backward_pass

def test_preprocessing():
    """
    Test 1: Preprocessing test.
    Verifies that vocabulary size is 82 and word2index round-trips hand-picked words.
    """
    # Verify vocabulary size
    assert len(vocab) == 82, f"Expected vocabulary size 82, got {len(vocab)}"
    
    # Test round-trip mapping for hand-picked words
    test_words = ['mary', 'bathroom.', 'went', 'hallway.']
    for word in test_words:
        assert word in word2index, f"Word '{word}' missing from word2index"
        idx = word2index[word]
        assert vocab[idx] == word, f"Round-trip failed for word '{word}'"
        
    print('PASS: test_preprocessing')

def test_softmax():
    """
    Test 2: Softmax stability and normalization test.
    Verifies output sums to 1.0 (within 1e-9) and uniform distribution for zero vector.
    """
    # Check sum to 1.0 for arbitrary input
    vec = np.array([1.5, -2.3, 10.0, 0.4])
    probs = softmax(vec)
    assert abs(np.sum(probs) - 1.0) < 1e-9, f"Softmax sum is {np.sum(probs)}, expected 1.0"
    
    # Check uniform distribution for zero vector
    zero_vec = np.zeros(5)
    uniform_probs = softmax(zero_vec)
    expected_prob = 1.0 / 5.0
    assert np.allclose(uniform_probs, expected_prob), 'Softmax on zero vector is not uniform'
    
    print('PASS: test_softmax')

def test_forward_shape():
    """
    Test 3: Forward pass output shape and untrained perplexity test.
    Verifies predict returns len(sent) + 1 layers and perplexity in [80, 84].
    """
    embed, recurrent, start, decoder, _ = init_weights(len(vocab))
    sent = words2indices(tokens[0][1:])
    
    layers, total_loss = predict(sent, embed, recurrent, start, decoder)
    
    # Verify layer count
    assert len(layers) == len(sent) + 1, f"Expected {len(sent) + 1} layers, got {len(layers)}"
    
    # Verify untrained perplexity range [80, 84]
    perplexity = compute_perplexity(total_loss, len(sent))
    assert 80.0 <= perplexity <= 84.0, f"Perplexity {perplexity:.2f} outside [80, 84]"
    
    print('PASS: test_forward_shape')

def test_training_decreases_loss():
    """
    Test 4: Fixed sentence training convergence test.
    Verifies that running 100 training steps on a single fixed sentence reduces perplexity.
    """
    embed, recurrent, start, decoder, one_hot = init_weights(len(vocab))
    sent = words2indices(tokens[0][1:])
    
    # Initial loss/perplexity
    _, initial_loss = predict(sent, embed, recurrent, start, decoder)
    initial_perp = compute_perplexity(initial_loss, len(sent))
    
    # Run 100 training steps on the same fixed sentence
    alpha = 0.01
    for _ in range(100):
        train_step(sent, embed, recurrent, start, decoder, one_hot, alpha=alpha)
        
    _, final_loss = predict(sent, embed, recurrent, start, decoder)
    final_perp = compute_perplexity(final_loss, len(sent))
    
    assert final_perp < initial_perp, f"Training failed to reduce perplexity: initial {initial_perp:.2f}, final {final_perp:.2f}"
    
    print('PASS: test_training_decreases_loss')

def test_hidden_delta_sum():
    """
    Test 5: BPTT hidden delta sum test.
    Verifies that for sentence length >= 3, intermediate hidden delta includes the 
    second component pulled through recurrent.T rather than output_delta alone.
    """
    embed, recurrent, start, decoder, one_hot = init_weights(len(vocab))
    sent = words2indices(tokens[0][1:])  # Sentence length >= 3
    
    # Perform forward pass
    layers, _ = predict(sent, embed, recurrent, start, decoder)
    
    # Perform backward pass to compute deltas without updating weights
    layers = backward_pass(sent, layers, decoder, recurrent, one_hot)
    
    # Direct component calculation for comparison
    target_idx = sent[0]
    out_delta = layers[1]['pred'].copy()
    out_delta[target_idx] -= 1.0
    direct_component = out_delta.dot(decoder.T)
    
    actual_hidden_delta = layers[1]['hidden_delta']
    
    # Verify hidden delta includes the recurrent backpropagation piece
    assert not np.allclose(actual_hidden_delta, direct_component), \
        'Hidden delta only contains output delta component; missing recurrent backpropagation piece!'
        
    print('PASS: test_hidden_delta_sum')

def main():
    # Run all test suites and output PASS/FAIL status.
    tests = [
        test_preprocessing,
        test_softmax,
        test_forward_shape,
        test_training_decreases_loss,
        test_hidden_delta_sum
    ]
    
    passed = 0
    failed = 0
    
    print('Running Visions Test Suite')
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} - {e}")
            failed += 1
            
    print(f"\nSummary: {passed} passed, {failed} failed.")

main()