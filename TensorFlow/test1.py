import argparse


def parse_args():
    description = "ok"

    parser = argparse.ArgumentParser(description=description)

    help = "ok1"
    parser.add_argument('addresses', nargs='*', help=help)

    help = "ok2"
    parser.add_argument('filename', help=help)

    help = "ok3"
    parser.add_argument('-p', '--port', type=int, help=help)

    help = "ok4"
    parser.add_argument('--iface', help=help) # default='localhost'

    help = "ok5"
    parser.add_argument('--delay', type=float, help=help, default=.7)

    help = "ok6"
    parser.add_argument('--bytes', type=int, help=help, default=10)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print args

    for address in args.addresses:
        print 'The address is : %s .' % address

    print 'The filename is : %s .' % args.filename
    print 'The port is : %d.' % args.port
    print 'The interface is : %s.' % args.iface
    print 'The number of seconds between sending bytes : %f' % args.delay
    print 'The number of bytes to send at a time : %d.' % args.bytes
