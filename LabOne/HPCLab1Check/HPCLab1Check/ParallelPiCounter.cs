using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;

namespace HPCLab1Check
{
    sealed internal class ParallelPiCounter
    {
        public int NumberOfWorkers { get; set; }
        public double Precision { get; set; }
        private readonly ConcurrentStack<double> _fractionsStack;

        public ParallelPiCounter()
        {
            _fractionsStack = new ConcurrentStack<double>();
        }

        public double CountPi()
        {
            if (Precision < NumberOfWorkers)
            {
                throw new Exception("Precision smaller than the number of processes - try again.");
            }

            Parallel.For(0, NumberOfWorkers, i =>
                                             {
                                                 var myPi = 0.0d;
                                                 var myIndex = i*2 + 1;
                                                 var mySign = ((myIndex - 1) / 2) % 2 == 1 ? -1 : 1;

                                                 while (myIndex < Precision)
                                                 {
                                                     Console.WriteLine("Worker {0}; Sign {1}; Index {2}", i, mySign, myIndex);
                                                     myPi += mySign/Convert.ToDouble(myIndex);
                                                     myIndex += 2*NumberOfWorkers;
                                                     mySign = ((myIndex - 1) / 2) % 2 == 1 ? -1 : 1;
                                                 }

                                                 _fractionsStack.Push(myPi);
                                             });

            var allFractions = _fractionsStack.ToArray().Sum();
            return allFractions*4;
        }
    }
}
