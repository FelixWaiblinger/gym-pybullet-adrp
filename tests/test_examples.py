def test_pid():
    from gym_pybullet_adrp.examples.pid import run
    run(gui=False, plot=False, output_folder='tmp')

def test_pid_velocity():
    from gym_pybullet_adrp.examples.pid_velocity import run
    run(gui=False, plot=False, output_folder='tmp')

def test_downwash():
    from gym_pybullet_adrp.examples.downwash import run
    run(gui=False, plot=False, output_folder='tmp')

def test_learn():
    from gym_pybullet_adrp.examples.learn import run
    run(gui=False, plot=False, output_folder='tmp', local=False)
