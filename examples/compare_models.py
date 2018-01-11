import tensorflow as tf
import time
import harness
import models


def compareModels():

    print "Long run"

    model_list = [models.simple0(), models.simple1(), models.simple2(), models.simple3(), models.simple4(), models.simple5(),
                  models.simple6(), models.simple7(), models.simple8(), models.simple9(), models.simple10(), models.simple11(),
                  models.simple12(), models.simple13(), models.simple14(), models.simple15(), models.simple16(), models.simple17(),
                  models.simple18(), models.simple19()]

    results = []

    for item in model_list:
        results += [item.__class__.__name__, harness.train_model_and_report(item, epochs=200, learning_rate_value=1e-3)]

    print """

  ====================================
        FINAL RESULTS
  ====================================

  """

    for item in results:
        print item

    print """
  RESULTS

  0.459933434452      simple18
  0.481827340846      simple9
  0.484590395482      simple11
  0.517686724076      simple19
  0.523299952174      simple10
  0.608307233759      simple16
  0.646958202629      simple15
  0.651675411848      simple8
  0.667441813173      simple7
  0.681077655134      simple14
  0.682826670507      simple5
  0.707398648726      simple6
  0.89458271703       simple13
  0.997465072211      simple12
  1.24220456685       simple3
  1.3567156149        simple2
  1.53155471589       simple4
  7.31041503028       simple1
  7.33090554099       simple0
  88.6500002875       simple17

  """

if __name__ == '__main__':
    compareModels()
