require 'rnn'
require 'optim'

batchSize = 30
rho = 20
hiddenSize = 512
hiddenSize1 = 256
inputSize = 400
outputSize = 4
epochs = 10000
xStart = 6
yStart = 2
yEnd = 5


local train_file_path = 'recurrent_train.th7' 
local train_data = torch.load(train_file_path)
local Y = train_data[{{},{yStart,yEnd}}]
local X = train_data[{{},{xStart,-1}}]
local place = train_data[{{},{1}}]
seriesSize = (#train_data)[1]
print(seriesSize)
local test_file_path = 'recurrent_test.th7'
local test_data = torch.load(test_file_path)
local test_labels = test_data[{{},{yStart,yEnd}}]
local test_X = test_data[{{},{xStart,-1}}]

model = nn.Sequential()
model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize, rho)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize, hiddenSize1, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize1, outputSize)))

criterion = nn.SequencerCriterion(nn.AbsCriterion())
--local method = 'xavier'
--local model_new = require('weight-init')(model, method)

-- define the index of the batch elements
offsets = {}
function offset_(seed)
offsets = {}
math.randomseed(seed)
for i= 1, batchSize do
   table.insert(offsets, math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)
end
function nextBatch()
   local inputs, targets = {}, {}
   local nums = {}
   for step = 1, rho do
      --get a batch of inputs
      table.insert(inputs, X:index(1, offsets))
      -- shift of one batch indexes
      offsets:add(1)
      for j=1,batchSize do
         if offsets[j] > seriesSize then
            offsets[j] = 1
         end
      end
      -- a batch of targets
      table.insert(targets, Y[{{},{1,4}}]:index(1,offsets))
      table.insert(nums,place:index(1,offsets))
   end
   return inputs, targets
end

-- get weights and loss wrt weights from the model
x, dl_dx = model:getParameters()

feval = function(x_new)
	-- copy the weight if are changed
	if x ~= x_new then
		x:copy(x_new)
	end

	-- select a training batch
	local inputs, targets = nextBatch()

	-- reset gradients (gradients are always accumulated, to accommodate
	-- batch methods)
	dl_dx:zero()

	-- evaluate the loss function and its derivative wrt x, given mini batch
	local prediction = model:forward(inputs)
	local loss_x = criterion:forward(prediction, targets)
	model:backward(inputs, criterion:backward(prediction, targets))
	
	return loss_x, dl_dx
end

adagrad_params = {
   learningRate = 0.01,
   learningRateDecay = 1e-08,
   weightDecay = 0,
   momentum = 0
}
seed = 1
offset_(seed)
time = sys.clock()
for j = 1, epochs do
	if j%1000 == 0 then 
		seed = seed + 1
		offset_(seed)
	end
   -- train a mini_batch of batchSize in parallel
	 _, fs = optim.adagrad(feval,x, adagrad_params)
	print('error for iteration ' .. adagrad_params.evalCounter  .. ' is ' .. fs[1]/rho)

end


print('id  approx   text')
local loss1 = 0.0
local loss2 = 0.0
local loss3 = 0.0
local loss4 = 0.0
predict_batch = 100
for i = 1,(#test_data)[1], predict_batch do
   local inputs = {}
   for step = 0, predict_batch-1 do
      --get a batch of inputs
      table.insert(inputs, test_X[i+step])
   end
   local myPrediction = model:forward(inputs)
   for step = 1, predict_batch do
   loss1 = loss1+math.abs(myPrediction[step][1] - test_labels[i+step-1][1])
   loss2 = loss2+math.abs(myPrediction[step][2] - test_labels[i+step-1][2])
   loss3 = loss3+math.abs(myPrediction[step][3] - test_labels[i+step-1][3])
   loss4 = loss4+math.abs(myPrediction[4] - test_labels[i][4])
   end

end

loss1 = loss1/(#test_data)[1]
loss2 = loss2/(#test_data)[1]
loss3 = loss3/(#test_data)[1]
loss4 = loss4/(#test_data)[1]

-- time taken
time = sys.clock() - time
print( "Time per epoch = " .. (time / epochs) .. '[s]')

print(loss1,loss2,loss3,loss4)
torch.save('recurrent3.dat',model)
