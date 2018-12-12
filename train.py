from torch.autograd import Variable
import time
import torch






def train_model(model, criterion, optimizer, dataloders, dataset_sizes, use_gpu, num_epochs=50,
                liste_save =[10, 20, 30, 40, 50, 60, 70]):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param dataloaders:
    :param dataset_sizes:
    :param use_gpu:
    :param num_epochs:
    :param liste_save:
    :return:
    """
    since = time.time()
    val_min = 100
    losses_train = []
    accuracy_train = []
    losses_val = []
    accuracy_val = []

    print(dataset_sizes)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_images', 'val_images']:
            if phase == 'train_images':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda().long())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                if labels.shape[1]==1:
                    labels = labels.squeeze(1)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train_images':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / float(dataset_sizes[phase])
                epoch_acc = float(running_corrects.data.item()) / float(dataset_sizes[phase])

                if phase == 'train_images':
                    losses_train.append(epoch_loss)
                    accuracy_train.append(epoch_acc)
                else:
                    losses_val.append(epoch_loss)
                    accuracy_val.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val_images' and epoch_acc > best_acc and val_min > epoch_loss:
                best_acc = epoch_acc
                val_min = epoch_loss
                best_model_wts = model.state_dict()
                state = {'model': best_model_wts, 'optim': optimizer.state_dict()}
                torch.save(state, 'experiment/point_resnet_best.pth')

        if epoch in liste_save:
            model_epoch = model.state_dict()
            state = {'model': model_epoch, 'optim': optimizer.state_dict()}
            torch.save(state, 'experiment/model_epoch_' + str(epoch)+ '.pth')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, losses_train, accuracy_train, losses_val, accuracy_val