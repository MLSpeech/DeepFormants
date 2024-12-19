require 'rnn'
require 'optim'

function range(from, to, step)
  step = step or 1
  return function(_, lastvalue)
    local nextvalue = lastvalue + step
    if step > 0 and nextvalue <= to or step < 0 and nextvalue >= to or
       step == 0
    then
      return nextvalue
    end
  end, nil, from - step
end

local train_file_path = 'recurrent_train.th7' 
local test_file_path = 'recurrent_test.th7'
local train_data = torch.load(train_file_path)
local test_data = torch.load(test_file_path)
local Y = train_data[{{},{2,5}}]
local X = train_data[{{},{6,-1}}]
local test_labels = test_data[{{},{2,5}}]
local test_X = test_data[{{},{6,-1}}]

batchSize = 5
rho = 10
hiddenSize1 = 1024
hiddenSize2 = 512
hiddenSize3 = 256
inputSize = 1
outputSize = 1
seriesSize = 100

model = nn.Sequential()
model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize2, rho)))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize2, hiddenSize3, rho)))
--model:add(nn.Sequencer(nn.Linear(hiddenSize2, hiddenSize3, rho)))
--model:add(nn.Sequencer(nn.Sigmoid()))
model:add(nn.Sequencer(nn.Linear(hiddenSize3, outputSize)))

criterion = nn.SequencerCriterion(nn.MSECriterion())

-- dummy dataset (task predict the next item)
--dataset = torch.randn(seriesSize, inputSize)

-- define the index of the batch elements
offsets = {}
for i= 1, batchSize do
   table.insert(offsets,i)
end
offsets = torch.LongTensor(offsets)
print(offsets)
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
      table.insert(targets, Y:index(1,offsets))
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

for i = 1, 2 do
   -- train a mini_batch of batchSize in parallel
   _, fs = optim.adagrad(feval,x, sgd_params)

  if sgd_params.evalCounter % 100 == 0 then
    print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
  end
end
