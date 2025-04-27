'''Implement Artificial Neural Network training process in Python by using Forward Propagation, 
Back Propagation.  '''
import numpy as np 

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_derv(x):
    return x * (1 - x)


#XOR
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])

input_layer=X.shape[1]
hidden_layer= 4
out_layer=1

wh=np.random.uniform(size=(input_layer , hidden_layer))
bh=np.random.uniform(size=(1,hidden_layer))

wo=np.random.uniform(size=(hidden_layer,out_layer))
bo=np.random.uniform(size=(1,out_layer))

epochs = 1
lr=0.01


for epoch in range (epochs):
    #Forward Propagation # hid_input is X
    hid_sum=np.dot(X,wh)+bh
    hid_output = sigmoid(hid_sum)  # output layer input

    final_sum = np.dot(hid_output,wo)+bo
    final_output= sigmoid(final_sum)

    #Backward Propagation
    error_out = y - final_output
    delta_out = error_out * sigmoid_derv(final_output)

    # error_hid = delta_output.dot(wo.T)
    error_hid = delta_out @ wo.T
    delta_hid = error_hid * sigmoid_derv(hid_output)

    wh += lr * X.T @ delta_hid
    bh+= np.sum(delta_hid, axis=0) * lr

    # wo+= lr + delta_output @ hid_output 
    wo = lr * hid_output.T @ delta_out
    bo+= np.sum(delta_out ,axis=0)* lr

    if epoch % 1000 ==0 :
        print( f"Error for epoch = {epoch} is {np.mean(error)}")


print(f"Final Prediction :{final_output}")
final_output= sigmoid(final_input)      

print(final_output)
    
