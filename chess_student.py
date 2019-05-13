import numpy as np
import matplotlib.pyplot as plt
from degree_freedom_queen import *
from degree_freedom_king1 import *
from degree_freedom_king2 import *
from features import *
from generate_game import *
from Q_values import *
import time
from tqdm import tqdm
from tqdm import tqdm_gui


size_board = 4


def main():
    """
    Generate a new game
    The function below generates a new chess board with King, Queen and Enemy King pieces randomly assigned so that they
    do not cause any threats to each other.
    s: a size_board   size_board matrix filled with zeros and three numbers:
    1 = location of the King
    2 = location of the Queen
    3 = location fo the Enemy King
    p_k2: 1x2 vector specifying the location of the Enemy King, the first number represents the row and the second
    number the colunm
    p_k1: same as p_k2 but for the King
    p_q1: same as p_k2 but for the Queen
    """
    s, p_k2, p_k1, p_q1 = generate_game(size_board)

    """
    Possible actions for the Queen are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right) multiplied by the number of squares that the Queen can cover in one movement which equals the size of 
    the board - 1
    """
    possible_queen_a = (s.shape[0] - 1) * 8
    """
    Possible actions for the King are the eight directions (down, up, right, left, up-right, down-left, up-left, 
    down-right)
    """
    possible_king_a = 8

    # Total number of actions for Player 1 = actions of King + actions of Queen
    N_a = possible_king_a + possible_queen_a

    """
    Possible actions of the King
    This functions returns the locations in the chessboard that the King can go
    dfK1: a size_board x size_board matrix filled with 0 and 1.
          1 = locations that the king can move to
    a_k1: a 8x1 vector specifying the allowed actions for the King (marked with 1): 
          down, up, right, left, down-right, down-left, up-right, up-left
    """
    dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Queen
    Same as the above function but for the Queen. Here we have 8*(size_board-1) possible actions as explained above
    """
    dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
    """
    Possible actions of the Enemy King
    Same as the above function but for the Enemy King. Here we have 8 possible actions as explained above
    """
    dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

    """
    Compute the features
    x is a Nx1 vector computing a number of input features based on which the network should adapt its weights  
    with board size of 4x4 this N=50
    """
    x = features(p_q1, p_k1, p_k2, dfK2, s, check)

    """
    Initialization
    Define the size of the layers and initialization
    FILL THE CODE
    Define the network, the number of the nodes of the hidden layer should be 200, you should know the rest. The weights 
    should be initialised according to a uniform distribution and rescaled by the total number of connections between 
    the considered two layers. For instance, if you are initializing the weights between the input layer and the hidden 
    layer each weight should be divided by (n_input_layer x n_hidden_layer), where n_input_layer and n_hidden_layer 
    refer to the number of nodes in the input layer and the number of nodes in the hidden layer respectively. The biases
     should be initialized with zeros.
    """
    n_input_layer = 50  # Number of neurons of the input layer. TODO: Change this value
    n_hidden_layer = 200  # Number of neurons of the hidden layer
    n_output_layer = 32  # Number of neurons of the output layer. TODO: Change this value accordingly

    """
    TODO: Define the w weights between the input and the hidden layer and the w weights between the hidden layer and the 
    output layer according to the instructions. Define also the biases.
    """

    w_input_hidden = np.random.rand(n_hidden_layer,n_input_layer)/(n_input_layer * n_hidden_layer)
    normW1 = np.sqrt(np.diag(w_input_hidden.dot(w_input_hidden.T)))
    normW1 = normW1.reshape(n_hidden_layer, -1)
    w_input_hidden = w_input_hidden/normW1
    
    w_hidden_output = np.random.rand(n_output_layer,n_hidden_layer)/(n_hidden_layer * n_output_layer)
    normW2 = np.sqrt(np.diag(w_hidden_output.dot(w_hidden_output.T)))
    normW2 = normW2.reshape(n_output_layer, -1)
    w_hidden_output = w_hidden_output/normW2
    
    bias_W1 = np.zeros((n_hidden_layer))
    bias_W2 = np.zeros((n_output_layer))


    # YOUR CODES ENDS HERE

    # Network Parameters
    epsilon_0 = 0.2   #epsilon for the e-greedy policy
    beta = 0.00005    #epsilon discount factor
    gamma = 0.85      #SARSA Learning discount factor
    eta = 0.0035      #learning rate
    N_episodes = 40000 #Number of games, each game ends when we have a checkmate or a draw
    alpha = 1/10000
    ###  Training Loop  ###

    # Directions: down, up, right, left, down-right, down-left, up-right, up-left
    # Each row specifies a direction, 
    # e.g. for down we need to add +1 to the current row and +0 to current column
    map = np.array([[1, 0],
                    [-1, 0],
                    [0, 1],
                    [0, -1],
                    [1, 1],
                    [1, -1],
                    [-1, 1],
                    [-1, -1]])
    
    # THE FOLLOWING VARIABLES COULD CONTAIN THE REWARDS PER EPISODE AND THE
    # NUMBER OF MOVES PER EPISODE, FILL THEM IN THE CODE ABOVE FOR THE
    # LEARNING. OTHER WAYS TO DO THIS ARE POSSIBLE, THIS IS A SUGGESTION ONLY.    

#    R_save = np.zeros([N_episodes, 1])
    R_save = np.zeros([N_episodes+1, 1])
    N_moves_save = np.zeros([N_episodes+1, 1])
    
    # END OF SUGGESTIONS
    

    for n in tqdm(range(N_episodes)):
#    for n in (range(N_episodes)):
        epsilon_f = epsilon_0 / (1 + beta * n) #psilon is discounting per iteration to have less probability to explore
        checkmate = 0  # 0 = not a checkmate, 1 = checkmate
        draw = 0  # 0 = not a draw, 1 = draw
        i = 1  # counter for movements

        # Generate a new game
        s, p_k2, p_k1, p_q1 = generate_game(size_board)

        # Possible actions of the King
        dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
        # Possible actions of the Queen
        dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
        # Possible actions of the enemy king
        dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
        
        
        Start = np.array([np.random.randint(size_board),np.random.randint(size_board)])   #random start
        s_start = np.ravel_multi_index(Start,dims=(size_board,size_board),order='F')      #conversion in single index
        s_index = s_start

        while checkmate == 0 and draw == 0:
            R = 0  # Reward

            # Player 1

            # Actions & allowed_actions
            a = np.concatenate([np.array(a_q1), np.array(a_k1)])
            allowed_a = np.where(a > 0)[0]
#            print(a)
#            print(allowed_a)
            # Computing Features
            x = features(p_q1, p_k1, p_k2, dfK2, s, check)

            # FILL THE CODE 
            # Enter inside the Q_values function and fill it with your code.
            # You need to compute the Q values as output of your neural
            # network. You can change the input of the function by adding other
            # data, but the input of the function is suggested. 
            
#            states_matrix = np.eye(size_board*size_board)
    
#            input_matrix = states_matrix[:,s_index].reshape((size_board*size_board),1)
            
            Q, out1 = Q_values(x, w_input_hidden, w_hidden_output, bias_W1, bias_W2)
#            print(Q)
#            print(np.argsort(-Q))
#            print(len(Q))
            """
            YOUR CODE STARTS HERE
            
            FILL THE CODE
            Implement epsilon greedy policy by using the vector a and a_allowed vector: be careful that the action must
            be chosen from the a_allowed vector. The index of this action must be remapped to the index of the vector a,
            containing all the possible actions. Create a vector called a_agent that contains the index of the action 
            chosen. For instance, if a_allowed = [8, 16, 32] and you select the third action, a_agent=32 not 3.
            """
            
            greedy = (np.random.rand() > epsilon_f)
            
            if greedy:
#                a_agent = np.random.choice(allowed_a)

                max_sort = np.argsort(-Q)
                for i in max_sort:
                    if i in allowed_a:
                        a_agent = i
                        break
                    else:
                        a_agent = np.random.choice(allowed_a)
#                if np.argmax(Q) in allowed_a:
#                    a_agent = np.argmax(Q)
#                else:
#                a_agent = np.argmax(Q)
                    
            else:
                a_agent = np.random.choice(allowed_a)
#                a_agent = a.index(a_agent)
                
#            if action in allowed_a:
#                a_agent = 

#            a_agent = 1  # CHANGE THIS VALUE BASED ON YOUR CODE TO USE EPSILON GREEDY POLICY
            
            #THE CODE ENDS HERE. 

#            print(a_agent)

            # Player 1 makes the action
            if a_agent < possible_queen_a:
                direction = int(np.ceil((a_agent + 1) / (size_board - 1))) - 1
                steps = a_agent - direction * (size_board - 1) + 1

                s[p_q1[0], p_q1[1]] = 0
                mov = map[direction, :] * steps
                s[p_q1[0] + mov[0], p_q1[1] + mov[1]] = 2
                p_q1[0] = p_q1[0] + mov[0]
                p_q1[1] = p_q1[1] + mov[1]

            else:
                direction = a_agent - possible_queen_a
                steps = 1

                s[p_k1[0], p_k1[1]] = 0
                mov = map[direction, :] * steps
                s[p_k1[0] + mov[0], p_k1[1] + mov[1]] = 1
                p_k1[0] = p_k1[0] + mov[0]
                p_k1[1] = p_k1[1] + mov[1]

            # Compute the allowed actions for the new position

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)

            # Player 2

            # Check for draw or checkmate
            if np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 1:
                # King 2 has no freedom and it is checked
                # Checkmate and collect reward
                checkmate = 1
                R = 1  # Reward for checkmate
                t = R + (gamma * max(Q))

                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, the agent gave checkmate.
                """

                deltaOut = (t-Q) * np.heaviside(Q, 0)
                w_hidden_output += eta * np.outer(deltaOut, out1)
                bias_W2 = eta * deltaOut
                
                deltaHid = np.dot(deltaOut,w_hidden_output) * np.heaviside(out1, 0)
                w_input_hidden = w_input_hidden + eta * np.outer(deltaHid, x)
                bias_W1 = eta * deltaHid
                
                R_save[n+1, 0] = alpha * R + (1-alpha) * R_save[n, 0]
                N_moves_save[n+1, 0] = alpha * i + (1-alpha) * N_moves_save[n, 0]
                # THE CODE ENDS HERE

                if checkmate:
                    break

            elif np.sum(dfK2) == 0 and dfQ1_[p_k2[0], p_k2[1]] == 0:
                # King 2 has no freedom but it is not checked
                draw = 1
                R = 0.1
#                print(Q)
                t = R + (gamma * max(Q))
                
                """
                FILL THE CODE
                Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
                rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
                the action made. You computed previously Q values in the Q_values function. Be careful: this is the last 
                iteration of the episode, it is a draw.
                """

                deltaOut = (t-Q) * np.heaviside(Q, 0)
                w_hidden_output += eta * np.outer(deltaOut, out1)
                bias_W2 = eta * deltaOut
                
                deltaHid = np.dot(deltaOut,w_hidden_output) * np.heaviside(out1, 0)
                w_input_hidden = w_input_hidden + eta * np.outer(deltaHid, x)
                bias_W1 = eta * deltaHid
                
                R_save[n+1, 0] = alpha * R + (1-alpha) * R_save[n, 0]
                N_moves_save[n+1, 0] = alpha * i + (1-alpha) * N_moves_save[n, 0]

                # YOUR CODE ENDS HERE
                

                if draw:
                    break

            else:
                # Move enemy King randomly to a safe location
                allowed_enemy_a = np.where(a_k2 > 0)[0]
                a_help = int(np.ceil(np.random.rand() * allowed_enemy_a.shape[0]) - 1)
                a_enemy = allowed_enemy_a[a_help]

                direction = a_enemy
                steps = 1

                s[p_k2[0], p_k2[1]] = 0
                mov = map[direction, :] * steps
                s[p_k2[0] + mov[0], p_k2[1] + mov[1]] = 3

                p_k2[0] = p_k2[0] + mov[0]
                p_k2[1] = p_k2[1] + mov[1]

            # Update the parameters

            # Possible actions of the King
            dfK1, a_k1, _ = degree_freedom_king1(p_k1, p_k2, p_q1, s)
            # Possible actions of the Queen
            dfQ1, a_q1, dfQ1_ = degree_freedom_queen(p_k1, p_k2, p_q1, s)
            # Possible actions of the enemy king
            dfK2, a_k2, check = degree_freedom_king2(dfK1, p_k2, dfQ1_, s, p_k1)
            # Compute features
            x_next = features(p_q1, p_k1, p_k2, dfK2, s, check)
            # Compute Q-values for the discounted factor
#            Q_next = Q_values(x_next, W1, W2, bias_W1, bias_W2)
            Q_next, demon = Q_values(x_next, w_input_hidden, w_hidden_output, bias_W1, bias_W2)
            t = R + (gamma * max(Q_next))
            """
            FILL THE CODE
            Update the parameters of your network by applying backpropagation and Q-learning. You need to use the 
            rectified linear function as activation function (see supplementary materials). Exploit the Q value for 
            the action made. You computed previously Q values in the Q_values function. Be careful: this is not the last 
            iteration of the episode, the match continues.
            """

            deltaOut = (t-Q) * np.heaviside(Q, 0)
            w_hidden_output += eta * np.outer(deltaOut, out1)
            bias_W2 = eta * deltaOut
                
            deltaHid = np.dot(deltaOut,w_hidden_output) * np.heaviside(out1, 0)
            w_input_hidden = w_input_hidden + eta * np.outer(deltaHid, x_next)
            bias_W1 = eta * deltaHid

            # YOUR CODE ENDS HERE
            i += 1
#        print(R)
        R_save[n+1, 0] = alpha * R + (1-alpha) * R_save[n, 0]
        N_moves_save[n+1, 0] = alpha * i + (1-alpha) * N_moves_save[n, 0]
    
    return R_save, N_moves_save
        
        


if __name__ == '__main__':
#    repetitions = 15
#    N_episodes = 1000
#    totalRewards = np.zeros((repetitions, N_episodes))
#    fontSize = 18
#    
#    for j in tqdm(range(repetitions)):
#        totalRewards[j,:] = main()
#        
#    plt.figure(figsize = (8, 6))
#    means = np.mean(totalRewards, axis = 0)
#    errors = 2 * np.std(totalRewards, axis = 0) / np.sqrt(repetitions) # errorbars are equal to twice standard error i.e. std/sqrt(samples)
#    plt.errorbar(np.arange(N_episodes), means, errors, 0, elinewidth = 2, capsize = 4, alpha =0.8)
#    plt.xlabel('Trial',fontsize = fontSize)
#    plt.ylabel('Average Reward',fontsize = fontSize)
#    plt.axis((-(N_episodes/10.0),N_episodes,-0.1,1.1))
#    plt.tick_params(axis = 'both', which='major', labelsize = 14)
#    plt.show()
    
    check, check2 = main()
#    plt.subplot(211)
#    
#    plt.ylabel('Average Reward',fontsize = 18)
#    plt.plot(check)
#    
#    plt.subplot(212)
#    plt.xlabel('Episode',fontsize = 18)
#    plt.show()
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(check)
    axarr[1].plot(check2)
    
