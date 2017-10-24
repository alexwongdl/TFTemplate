"""
Created by Alex Wang
On 2017-10-24
"""
def arg_parse_print(FLAGS):
    """
    FLAGS = parser.parse_args()
    :param FLAGS:
    :return:
    """
    print('[Configurations]:')
    for name in FLAGS.__dict__.keys():
        value = FLAGS.__dict__[name]
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('[End of configuration]')