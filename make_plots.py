import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt






def load_data():
	data_sets = {}

	data_fnames = os.listdir('results/')

	for fname in data_fnames:
		env = fname[:fname.find('_')].replace('CT', '')
		if env not in data_sets:
			data_sets[env] = []

		tmp_data = np.genfromtxt(os.path.join('results', fname), delimiter=',')

		data_sets[env].append(tmp_data)

	return data_sets







def main():
	data_sets = load_data()
	print(data_sets.keys())

	envs = ['Hopper', 'Reacher', 'InvertedPendulum']

	gammas = ['1', '0.99']

	i = 0
	for env in envs:
		

		env = 'Roboschool' + env + '-v1'
		PPO = np.mean(np.array(data_sets[env]),axis = 0)

		time_env = list(env)
		time_env.insert(env.find('-'), 'Limited')
		time_env = ''.join(time_env)
		print(time_env)

		PPO_time = np.array(data_sets[time_env])
		PPO_time = np.mean(PPO_time, axis= 0)

		print(PPO_time.shape)

		for gamma in gammas:

			y = 1
			if gamma == '0.99':
				y = 2


			PPO_x = PPO[:,0]
			PPO_y = PPO[:,y]

			PPO_time_x = PPO_time[:,0]
			PPO_time_y = PPO_time[:,y]

			plt.close()

			plt.plot(PPO_x, PPO_y, 'g', label='PPO')
			plt.plot(PPO_time_x, PPO_time_y, 'b', label='PPO_time')

			plt.xlabel('timesteps')
			plt.ylabel('returns (gamma = ' + gamma + ')')

			plt.legend()



			plt.savefig(os.path.join('plots', env+gamma+'.png'))


		






















if __name__ == '__main__':
	main()


