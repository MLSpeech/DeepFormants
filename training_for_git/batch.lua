require 'torch'   -- torch
require 'optim'
require 'nn'      -- provides a normalization operator
local train_file_path = 'train.th7' 
local test_file_path = 'test.th7'
local train_data = torch.load(train_file_path)
local test_data = torch.load(test_file_path)
local Y = train_data[{{},{2,5}}]
local X = train_data[{{},{6,-1}}]
local test_labels = test_data[{{},{2,5}}]
local test_X = test_data[{{},{6,-1}}]
local batch_size = 30
epochs = 3

model = nn.Sequential()                 -- define the container
ninputs = 350; noutputs = 4 ; nhiddens1 = 1024; nhiddens2 = 512; nhiddens3 = 256
model:add(nn.Linear(ninputs,nhiddens1))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens1,nhiddens2))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens2,nhiddens3))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens3,noutputs))
criterion = nn.AbsCriterion()--MSECriterion()
x, dl_dx = model:getParameters()
sgd_params = {
   learningRate = 0.01,
   learningRateDecay = 1e-08,
   weightDecay = 0,
   momentum = 0
}

function train(X,Y)
	   
   current_loss = 0
   for batch = 1,(#train_data)[1], batch_size do

      local inputs = {}
      local targets = {} 
      local x_start = batch
      local x_end = math.min(batch + batch_size-1, (#train_data)[1])
      for i = x_start,x_end do    
      local target = Y[i]    
      local input = X[i]  
      table.insert(inputs, input)
      table.insert(targets, target)
      end
	local feval = function(x_new)
	   if x ~= x_new then
	      x:copy(x_new)
	   end
	   dl_dx:zero()
           local f=0
	for i = 1, #inputs do
	   local loss_x = criterion:forward(model:forward(inputs[i]), targets[i])
	   model:backward(inputs[i], criterion:backward(model.output, targets[i]))
	f = f+loss_x
	end
	   return f/#inputs, dl_dx:div(#inputs)
	end
      _,fs = optim.adagrad(feval,x,sgd_params)
      current_loss = current_loss + fs[1]
   end
   current_loss = current_loss/( (#train_data)[1]/batch_size)
   print('train loss = ' .. current_loss)
   return current_loss
end

time = sys.clock()
local cumm_loss = 0.
for j = 1, epochs do
    print(j)
    cumm_loss = train( X, Y )
print( 'Final loss = ' .. cumm_loss )
if j%10 == 0 then
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
end
end


-- time taken
time = sys.clock() - time
print( "Time per epoch = " .. (time / epochs) .. '[s]')



print(loss1,loss2,loss3,loss4)
torch.save('estimation_model.dat',model)
