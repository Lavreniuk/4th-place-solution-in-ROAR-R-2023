/***********[hash.cc]
Copyright (c) 2012-2013 Jessica Davies, Fahiem Bacchus

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

***********/
#include <iostream> 
#include "maxhs/utils/hash.h"

uint32_t rnd_table[256] {
  868768155,
  618522103,
  1552882209,
  80221016,
  115080558,
  1055832139,
  1667253458,
  1049170342,
  1376460154,
  914249860,
  1048465347,
  1798137010,
  1336748788,
  23084515,
  1572777182,
  1617553847,
  1294164067,
  436165975,
  627889307,
  1696589762,
  110522491,
  1640277560,
  1505647036,
  2035382293,
  1236871088,
  1071825181,
  331719265,
  539832202,
  382784115,
  875153970,
  2028029813,
  1441390329,
  1881539726,
  1727691473,
  28040939,
  1878261955,
  1412035312,
  876827580,
  1481410519,
  670824079,
  371938526,
  90214601,
  1827273279,
  354440472,
  1484616607,
  1759502545,
  1008161564,
  2001922703,
  1356933857,
  1321590409,
  590142944,
  1748799407,
  330173733,
  1574583625,
  933847728,
  253430945,
  1298634662,
  631020000,
  2086556703,
  1549606358,
  753578192,
  852859565,
  1501832716,
  343769414,
  703022035,
  1081351353,
  517752345,
  1318618932,
  440881018,
  737017595,
  602960354,
  219203506,
  1412083669,
  430347331,
  23530358,
  332508514,
  1291241068,
  2042008766,
  1961739732,
  615337210,
  1467371089,
  1558149855,
  1933891998,
  1731832553,
  334417547,
  1053999114,
  446362737,
  728685744,
  1428390738,
  628800086,
  711439712,
  1533611090,
  2051997828,
  1420968132,
  61110743,
  1396607964,
  1184387835,
  1040299241,
  447856852,
  1920563169,
  356411821,
  1076025289,
  7174075,
  35064789,
  571032709,
  921053058,
  753423852,
  798435210,
  540736423,
  974030699,
  995027592,
  1152853982,
  299300735,
  1546724501,
  1166822243,
  2025907870,
  972313400,
  60442387,
  609713336,
  1122814210,
  1053310414,
  786974038,
  2054067268,
  1711537086,
  1403090724,
  2079619923,
  1994748628,
  1321869068,
  750557700,
  1816573534,
  995375555,
  1679222171,
  1033847202,
  2107300677,
  1405887001,
  145780680,
  2005813479,
  255280532,
  108724045,
  1898986685,
  32611369,
  822157357,
  1298063381,
  539756210,
  1015967583,
  1276450022,
  1989604415,
  2133683542,
  1806888482,
  180083151,
  1702184492,
  759877507,
  964764915,
  2074884322,
  113540263,
  374289229,
  1204999757,
  1556517059,
  2045067204,
  860774250,
  1699820024,
  1678611466,
  1623108508,
  582914510,
  1021387745,
  1771689028,
  1578106011,
  888634528,
  1760316151,
  1582473338,
  1691490323,
  2065797934,
  1761563298,
  2028511240,
  1841907120,
  424510328,
  75534326,
  520456178,
  1807022906,
  2129606240,
  2018509180,
  598140664,
  137481233,
  1857742107,
  1368716682,
  1222296034,
  1322810350,
  2105305561,
  2100126602,
  1370822212,
  1722231619,
  1747309463,
  1832908485,
  1581620306,
  68894331,
  1702316815,
  523706985,
  581414739,
  1656935161,
  1658415612,
  1950184377,
  49617129,
  100496977,
  867482541,
  1597118940,
  400033582,
  1351334762,
  18196963,
  1396548911,
  440200899,
  51571471,
  1514057376,
  513077239,
  519979830,
  2115280706,
  258504016,
  569855695,
  674587387,
  1830092354,
  477778202,
  298450372,
  716648642,
  1030304697,
  1505035619,
  1473749293,
  1939987286,
  2033186565,
  939405941,
  730447859,
  1302150910,
  1556026393,
  946567952,
  678335411,
  1820823012,
  322717758,
  1838426029,
  1043439820,
  1525087374,
  892977192,
  1441316008,
  1108755771,
  948521121,
  900305804,
  543609543,
  1473238656,
  861457369,
  1533572544,
  1931607662,
  1803982569,
  1920017701,
  1255070917,
  1178515439,
  758426408,
  226456279,
  1986288478,
  2058055390
    };

void BinClsTable::insert(Lit l1, Lit l2) {
  auto key = mkKey(l1, l2);
  auto& v = bins[key];
  if(var(l1) > var(l2))
    swap(l1, l2);
  v.push_back(l1);
  v.push_back(l2);
}

vector<Lit>* BinClsTable::get(Var v1, Var v2) {
  auto key = mkKey(v1, v2);
  auto it = bins.find(key);
  if(it == bins.end())
    return nullptr;
  return &(it->second);
}