xshift = -0.5;
yshift = 0.05;

lc1 = 0.2;

Point(1) = {0.0,0,0,lc1};

r = 0.3;
Point(2) = {r,0,0,lc1};
Point(3) = {0.0,r,0,lc1};
Point(4) = {-r,0,0,lc1};
Point(5) = {0,-r,0,lc1};

Circle(1) = {2,1,3};
Circle(2) = {3,1,4};
Circle(3) = {4,1,5};
Circle(4) = {5,1,2};

Line Loop(5) = {1,2,3,4};

//+
Translate {1.5, 0, 0} {
  Line{2}; Line{3}; Line{1}; Line{4}; 
}

//+
Rotate {{0, 0, 1}, {0, 0, 0}, 2*Pi/3} {
  Duplicata { Line{2}; Line{3}; Line{1}; Line{4}; }
}
//+
Rotate {{0, 0, 1}, {0, 0, 0}, -2*Pi/3} {
  Duplicata { Line{2}; Line{3}; Line{4}; Line{1}; }
}


R = 5; i = 1000; lc2=0.8;

Point(1+i) = {0.0,0,0, lc2};
Point(2+i) = {R,0,0, lc2};
Point(3+i) = {0.0,R,0, lc2};
Point(4+i) = {-R,0,0, lc2};
Point(5+i) = {0,-R,0, lc2};

Circle(1+i) = {2+i,1+i,3+i};
Circle(2+i) = {3+i,1+i,4+i};
Circle(3+i) = {4+i,1+i,5+i};
Circle(4+i) = {5+i,1+i,2+i};

Line Loop(5+i) = {1+i,2+i,3+i,4+i}; 


//+
Line Loop(1006) = {8, 6, 7, 9};
//+
Line Loop(1007) = {11, 12, 13, 10};
//+
Plane Surface(1) = {5, 1005, 1006, 1007};

Mesh 2;
Coherence Mesh;

