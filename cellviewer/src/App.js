import React, { Component } from 'react';
import { Route, Switch } from 'react-router-dom';
import Viewer from './components/Viewer/Viewer';

class App extends Component {

    render() {
        return (
            <Switch>
                <Route exact path="/" component={Viewer} />
            </Switch>
        );
    }
}

export default App;
