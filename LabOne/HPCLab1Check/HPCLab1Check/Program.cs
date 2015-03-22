using System;

namespace HPCLab1Check
{
    class Program
    {
        static void Main()
        {
            var counter = new ModifiedParallelPiCounter()
                          {
                              NumberOfWorkers = 10,
                              Precision = 10
                          };

            Console.WriteLine("Result is {0}", counter.CountPi());
            Console.ReadLine();

            //for (var myIndex = 1; myIndex < 100; myIndex += 2*3)
            //{
            //    var mySign = ((myIndex - 1)/2)%2 == 0 ? -1 : 1;
            //    Console.WriteLine("Sign - {0}", mySign);
            //}

            //Console.ReadLine();
        }
    }
}
