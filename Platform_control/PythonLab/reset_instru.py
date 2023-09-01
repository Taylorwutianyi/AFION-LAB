import time

ul = 1e-6

def reset_pumps(p,op):
    print ("reset pumps\n")

    # p.dispense_all()
    op.dispense_all()

    p.init_syringe(init_valve = 1, syringe_volume = 100 * ul, psdtype = 'psd8')
    op.init_syringe(init_valve = 8, syringe_volume = 2500 * ul, psdtype = 'psd6')
    op.set_velocity(op.get_max_steps() * 2 // 70)

    op.draw(volume=2400 * ul)
    p.draw_and_dispense(8, 1, 250*ul,velocity=1000)
    p.draw_and_dispense(7, 1, 450*ul,velocity=1000)
    time.sleep(3)

    op.dispense_all()
    op.draw(volume=2400 * ul)
    time.sleep(3)

def reset_UV(ard):
    print ("reset UV\n")
    ard.turnoff()
