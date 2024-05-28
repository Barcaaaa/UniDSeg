import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({'font.size': 18})

x = ["25", "50", "100", "150", "200", "300"]
y1 = [67.1, 68.0, 68.2, 68.2, 67.3, 66.7]
y2 = [41.5, 42.1, 43.2, 42.8, 41.7, 41.5]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

ax1.grid(True)
ax1.set_axisbelow(True)
line1, = ax1.plot(x, y1, marker='*', color='darkmagenta', label='USA/Sing.', linewidth=3, markersize=20)
ax1.set_ylim([66, 70])
ax1.set_xticklabels([])

ax2.grid(True)
ax2.set_axisbelow(True)
line2, = ax2.plot(x, y2, marker='s', color='royalblue', label='A2D2/sKITTI', linewidth=3, markersize=15)
ax2.set_ylim([41, 45])
ax2.tick_params(axis='x', labelsize=18)

for i, v in enumerate(y1):
    ax1.text(i, v + 0.3, str(v), ha='center', va='bottom')
for i, v in enumerate(y2):
    ax2.text(i, v + 0.3, str(v), ha='center', va='bottom')

ax1.set_ylabel('2D mIoU (%)')
ax2.set_xlabel('Token Length')
ax2.set_ylabel('2D mIoU (%)')

lines = [line1, line2]
ax2.legend(lines, [line.get_label() for line in lines], loc='upper right')

# plt.savefig('figure.png')
plt.savefig('abs_token.pdf', format='pdf')

