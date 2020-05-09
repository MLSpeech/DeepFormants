require 'torch'   -- torch
require 'optim'
require 'nn'      -- provides a normalization operator
local train_file_path = 'train.th7' 
local test_file_path = 'test.th7'
local train_data = torch.load(train_file_path)
local test_data = torch.load(test_file_path)
local train_labels = train_data[{{},{2,5}}]
local train_X = train_data[{{},{6,-1}}]
local test_labels = test_data[{{},{2,5}}]
local test_X = test_data[{{},{6,-1}}]
local batch_size = 30
model = nn.Sequential()                 -- define the container
ninputs = 350; noutputs = 4 ; nhiddens1 = 1024; nhiddens2 = 512; nhiddens3 = 256
--model:add(nn.Linear(ninputs, noutputs)) -- define the only module
model:add(nn.Linear(ninputs,nhiddens1))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens1,nhiddens2))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens2,nhiddens3))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens3,noutputs))
criterion = nn.AbsCriterion()--MSECriterion()
x, dl_dx = model:getParameters()

feval = function(x_new)
   if x ~= x_new then
      x:copy(x_new)
   end
   -- select a new training sample
   _nidx_ = (_nidx_ or 0) + 1
   if _nidx_ > (#train_data)[1] then _nidx_ = 1 end
   --local sample = data[_nidx_]
   local target = train_labels[_nidx_]      -- this funny looking syntax allows
   local inputs = train_X[_nidx_]    -- slicing of arrays.
   -- reset gradients (gradients are always accumulated, to accommodate 
   -- batch methods)
   dl_dx:zero()
   -- evaluate the loss function and its derivative wrt x, for that sample
   --print(inputs)
   --print(target)
   for i=1, 350 do
  if type(inputs[i]) ~= 'number' then
  print(i)
  print(inputs[i])
  print(type(inputs[i])) end
  end
   --io.write("continue with this operation (y/n)?")
   --answer=io.read()
   local loss_x = criterion:forward(model:forward(inputs), target)
   model:backward(inputs, criterion:backward(model.output, target))
   -- return loss(x) and dloss/dx
   return loss_x, dl_dx
end
-- Given the function above, we can now easily train the model using SGD.
-- For that, we need to define four key parameters:
--   + a learning rate: the size of the step taken at each stochastic 
--     estimate of the gradient
--   + a weight decay, to regularize the solution (L2 regularization)
--   + a momentum term, to average steps over time
--   + a learning rate decay, to let the algorithm converge more precisely
sgd_params = {
   learningRate = 0.01,
   learningRateDecay = 1e-08,
   weightDecay = 0,
   momentum = 0
}
-- We're now good to go... all we have left to do is run over the dataset
-- for a certain number of iterations, and perform a stochastic update 
-- at each iteration. The number of iterations is found empirically here,
-- but should typically be determinined using cross-validation.
-- we cycle 1e4 times over our training data
for i = 1,1 do
   print(i)
   -- this variable is used to estimate the average loss
   current_loss = 0
   -- an epoch is a full loop over our training data
   for i = 1,(#train_data)[1] do
      -- optim contains several optimization algorithms. 
      -- All of these algorithms assume the same parameters:
      --   + a closure that computes the loss, and its gradient wrt to x, 
      --     given a point x
      --   + a point x
      --   + some parameters, which are algorithm-specific      
      _,fs = optim.adagrad(feval,x,sgd_params)
      -- Functions in optim all return two things:
      --   + the new x, found by the optimization method (here SGD)
      --   + the value of the loss functions at all points that were used by
      --     the algorithm. SGD only estimates the function once, so
      --     that list just contains one value.
      current_loss = current_loss + fs[1]
   end
   -- report average error on epoch
   current_loss = current_loss / (#train_data)[1]
   print('train loss = ' .. current_loss)
   
end
----------------------------------------------------------------------
-- 5. Test the trained model.

-- Now that the model is trained, one can test it by evaluating it
-- on new samples.

-- The text solves the model exactly using matrix techniques and determines
-- that 
--   corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

-- We compare our approximate results with the text's results.

print('id  approx   text')
local loss1 = 0.0
local loss2 = 0.0
local loss3 = 0.0
local loss4 = 0.0
for i = 1,(#test_data)[1] do
   local myPrediction = model:forward(test_X[i])
   loss1 = loss1+math.abs(myPrediction[1] - test_labels[i][1])
   loss2 = loss2+math.abs(myPrediction[2] - test_labels[i][2])
   loss3 = loss3+math.abs(myPrediction[3] - test_labels[i][3])
   loss4 = loss4+math.abs(myPrediction[4] - test_labels[i][4])
end

loss1 = loss1/(#test_data)[1]
loss2 = loss2/(#test_data)[1]
loss3 = loss3/(#test_data)[1]
loss4 = loss4/(#test_data)[1]

print(loss1,loss2,loss3,loss4)
torch.save('save.dat',model)
