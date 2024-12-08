require 'rnn'
require 'optim'

batchSize = 30
rho = 10
hiddenSize = 512
hiddenSize1 = 256
inputSize = 400
outputSize = 3
epochs = 100
xStart = 6
yStart = 2
yEnd = 4


local train_file_path = 'recurrent_train.th7' 
local train_data = torch.load(train_file_path)
local Y = train_data[{{},{yStart,yEnd}}]
local X = train_data[{{},{xStart,-1}}]
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

-- dummy dataset (task predict the next item)
--dataset = torch.randn(seriesSize, inputSize)

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets, i)--math.ceil(math.random() * batchSize))
end
offsets = torch.LongTensor(offsets)

function nextBatch()
   local inputs, targets = {}, {}
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
      table.insert(targets, Y[{{},{1,3}}]:index(1,offsets))
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

sgd_params = {
   learningRate = 0.01,
   learningRateDecay = 1e-08,
   weightDecay = 0,
   momentum = 0
}

time = sys.clock()
for j = 1, epochs do
   -- train a mini_batch of batchSize in parallel
	 _, fs = optim.adagrad(feval,x, sgd_params)
	print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1])

end


print('id  approx   text')
local loss1 = 0.0
local loss2 = 0.0
local loss3 = 0.0
local loss4 = 0.0
for i = 1,(#test_data)[1], 1 do
   local inputs = {}
   for step = 1, 1 do
      --get a batch of inputs
      table.insert(inputs, test_X[i])
   end
   local myPrediction = model:forward(inputs)
   loss1 = loss1+math.abs(myPrediction[1][1] - test_labels[i][1])
   loss2 = loss2+math.abs(myPrediction[1][2] - test_labels[i][2])
   loss3 = loss3+math.abs(myPrediction[1][3] - test_labels[i][3])
   --loss4 = loss4+math.abs(myPrediction[4] - test_labels[i][4])
end

loss1 = loss1/(#test_data)[1]
loss2 = loss2/(#test_data)[1]
loss3 = loss3/(#test_data)[1]
--loss4 = loss4/(#test_data)[1]

-- time taken
time = sys.clock() - time
print( "Time per epoch = " .. (time / epochs) .. '[s]')

print(loss1,loss2,loss3,loss4)
torch.save('recurrent.dat',model)
