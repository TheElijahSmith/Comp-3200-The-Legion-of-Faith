
# Architectures

### Wider

layer_0 --[ weights_0_1 (3, 32), relu ] --> layer_1 --[ weights_1_2 (32, 1) ] --> layer_2
 (1, 3)                                     (1, 32)                             (1,1)

Weights: 3x32 + 32x1 = 128

Rationale: This model would be excellent at capturing all of the possible combinations of interactions between the inputs and memorizing the data. This wastes a lot of compute, and due to having only one activation function, would struggle to capture high-dimensional relationships

### Deeper

layer_0 --[ weights_0_1 (3,4), relu ] --> layer_1 --[ weights_1_2 (4, 4), relu ] --> layer_2 --[ weights_2_3 (4, 4), relu ] -->
(1, 3)                                      (1, 4)                                      (1,4)
layer_3 --[ weights_3_4 (4, 1) ] --> layer_4
(1, 4)                                (1, 1)

Weights: 3x4 + 4x4 + 4x4 + 4x1 = 48

Rationale: This model is excellent for capturing very high-dimensional relationships while also being very efficient with compute. Some issues can arise with this model due to the narrow hidden layer size because it may miss some specific combinations of inputs and that will cause issues for those combinations.

### Multi-Output

layer_0 --[ weights_0_1 (3, 8), relu ] --> layer_1 --[ weights_1_2 (8, 4) ] --> layer_2
 (1, 3)                                     (1, 8)                             (1,4)

Weights: 3x8 + 8x4 = 56

Rationale: This model is good at capturing simpler, lower dimensional relationships between the inputs and outputs. It will struggle if any of the outputs require more complex computations, as the model cannot capture much of a higher dimensional relationship or simply more of the possible combinations of layer values.

### My Design

Given basic team statistics (yds per game, yds allowed, win percentage, avg game score) for both teams, predict final scores for both teams.

layer_0 --[ weights_0_1 (8, 32), relu ] --> layer_1 --[ weights_1_2 (32, 32), relu ] -->
(1, 8)                                      (1, 32)             
layer_2 --[ weights_2_3 (32, 2) ] --> layer_3
(1, 32)                                 (1, 2)

Weights: 8x32 + 32x32 + 32x2 = 1344

Rationale: This model is good at taking several inputs and evaluating many possible combinations of them, as well as capturing a decent amount of non-linearity. This model requires a decent amount of compute and may still struggle on very complex relationships.