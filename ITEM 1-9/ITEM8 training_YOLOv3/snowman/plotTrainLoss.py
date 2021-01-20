import matplotlib.pyplot as plt
 
log='train.log'
lines = []
for line in open(log):
    if "avg" in line:
        lines.append(line)
 
iterations = []
avg_loss = []
 
print('Retrieving data and plotting training loss graph...')
for i in range(len(lines)):
    try:
        lineParts = lines[i].split(',')
        iterations.append(int(lineParts[0].split(':')[0]))
        avg_loss.append(float(lineParts[1].split()[0]))
    except:
        continue
 
fig = plt.figure()
for i in range(0, len(lines)):
    plt.plot(iterations[i:i+2], avg_loss[i:i+2], 'r.-')
 
plt.xlabel('Batch Number')
plt.ylabel('Avg Loss')
fig.savefig('training_loss_plot.png', dpi=300)
 
print('Done! Plot saved as training_loss_plot.png')