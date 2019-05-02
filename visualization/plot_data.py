import numpy as np
import matplotlib.pyplot as plt

# # plot points
# plt.plot([1,2,3,4], [1,4,9,16], 'ro')

# # evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# # red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'b--', t, t**3, 'g^')
# plt.show()

# plot the runtime graph
cpu, = plt.loglog([10, 100, 1000, 10000], 
		[0.0042, 0.0524,  2.312, 224.6414],
		'ro-') 
gpu, = plt.loglog([10, 100, 1000, 10000, 100000], 
		[0.1204, 0.1326, 0.396, 2.8516, 44.5028],
		'bo-') 

plt.legend([cpu, gpu], ["CPU", "GPU"], loc='lower right')
plt.xlabel('Number of Nodes in Graph')
plt.ylabel('Runtime (s) ')
plt.title(' Loglog Plot of Runtimes for Various Graph Generation Sizes')
plt.show()


# plot the threshold graph
# plot the runtime graph
cpuT, = plt.loglog([10, 100, 1000, 10000], 
		[ 2380.95, 1908.40,  432.53 , 44.51  ],
		'ro-') 
gpuT, = plt.loglog([10, 100, 1000, 10000, 100000], 
		[83.06, 754.15,  2525.25, 3506.80, 2247.05],
		'bo-') 

plt.legend([cpuT, gpuT], ["CPU", "GPU"], loc='lower right')
plt.xlabel('Number of Nodes in Graph')
plt.ylabel('Throughput (Num Nodes / s) ')
plt.title('Loglog Plot of Throughputs for Various Graph Generation Sizes')
plt.show()


# plot the threshold graph
# plot the runtime graph
cpuT, = plt.loglog([10, 100, 1000, 10000], 
		[ 0.035, 0.395, 5.84, 78.82 ],
		'go-') 

plt.legend([cpuT], ["Speedup"], loc='lower right')
plt.xlabel('Number of Nodes in Graph')
plt.ylabel('Speedup (Runtime CPU / Runtime GPU) ')
plt.title('Loglog Plot of Speedup for Various Graph Generation Sizes')
plt.show()


