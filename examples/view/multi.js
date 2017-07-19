var garden = angular.module('garden' , []);


garden.controller('gardenController', function ($scope,$http,$timeout,$interval) {

    $scope.data = {
    	'images' : [],
        'training_sessions' : [0,0,0,0,0,0,0]
    }

    $scope.randomizeImage = function() {
        $scope.data.images = []
        for (index=0 ; index < 10 ; index++ ) {
            $scope.data.images.push( Math.floor( Math.random() * 10000 ) )
        }
    }

    $scope.resetSession = function() {
        $http({method:"GET" , url : "/resetSession" , cache: false}).then(function successCallback(result) {
            console.log("reset done");
            $scope.randomizeImage();
        })
    }

    $scope.learn = function(index) {
        console.log("LEARNING...");
        $http({ method : "GET" , url : "/learn/"+index , cache: false}).then(function successCallback(result) {
            console.log("...DONE");
            $scope.data.training_sessions[index] += 1;
            $scope.randomizeImage();

            if ($scope.data.training_sessions[index] % 5 != 0) {
                setTimeout( function() { $scope.learn(index) } , 5000 );
            }
        })
    }

});